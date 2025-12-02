# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
FastPersist save implementation without patching torch, using the
**non-legacy zipfile path**.

This module extends DeepSpeed's FastFileWriter to support scatter writes
into the zipfile format created by torch.serialization.skip_data().
"""

import os
import io
import time
import pickle
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from deepspeed.io.fast_file_writer import FastFileWriter, FastFileWriterConfig
from deepspeed.io.double_io_buffer import Double_IO_Buffer
from deepspeed.io.single_io_buffer import Single_IO_Buffer
from deepspeed.io.base_io_buffer import Base_IO_Buffer
from deepspeed.ops.op_builder import AsyncIOBuilder, GDSBuilder, UtilsBuilder
from deepspeed.accelerator import get_accelerator


class FastPersistScatterWriter:
    """
    Extends FastFileWriter's I/O infrastructure for scatter writes.
    
    FastFileWriter writes sequentially (append-only). This class uses the same
    Double_IO_Buffer and async_pwrite infrastructure but supports writing to
    arbitrary file offsets - required for the zipfile format.
    """
    
    def __init__(self, file_path: str, config: FastFileWriterConfig):
        self._file_path = file_path
        self._dnvme_handle = config.dnvme_handle
        self._double_buffer = config.double_buffer
        
        # Open file for read/write at arbitrary offsets (not append)
        self._fd = os.open(file_path, os.O_RDWR | os.O_DIRECT)
        
        # Use FastPersist's I/O buffer infrastructure
        io_buffer_type = Double_IO_Buffer if config.double_buffer else Single_IO_Buffer
        self._io_buffer = io_buffer_type(config.pinned_tensor, self._dnvme_handle)
        
        # For converting tensors to byte tensors
        self._cast_to_byte_tensor = UtilsBuilder().load().cast_to_byte_tensor
        
        # Statistics (matching FastFileWriter format)
        self._stats = {
            'close': 0,
            'fileno': 0,
            'flush': 0,
            'write': 0,
            'bytes': 0,
            'write_secs': 0.0,
            'aio_write_secs': 0.0,
            'aio_bytes': 0,
            'slow_bytes': 0,
            'slow_write_secs': 0.0,
            'fill_buffer_count': 0,
            'fill_buffer_secs': 0.0,
            'save_storage': 0,
            'save_storage_bytes': 0,
            'skeleton_secs': 0.0,
            'path': file_path,
        }
    
    def save_storages_at_offsets(
        self,
        storages: Dict[str, torch.UntypedStorage],
        offsets: Dict[str, int],
        use_gds: bool = False,
    ) -> int:
        """
        Write storages to their designated file offsets using FastPersist I/O.
        
        This is the scatter-write equivalent of FastFileWriter.save_torch_storage_object_list().
        """
        write_start = time.time()
        
        # Build sorted list of (offset, key, storage, nbytes)
        write_list: List[Tuple[int, str, torch.UntypedStorage, int]] = []
        for key, storage in storages.items():
            offset = offsets[key]
            nbytes = storage.nbytes()
            
            # Move to appropriate device
            if not use_gds and storage.device.type != "cpu":
                storage = storage.cpu()
            elif use_gds and storage.device.type != get_accelerator().device_name():
                storage = storage.to(get_accelerator().current_device_name())
            
            write_list.append((offset, key, storage, nbytes))
        
        # Sort by file offset for sequential-ish I/O pattern
        write_list.sort(key=lambda x: x[0])
        
        # Track batching state
        batch_start_offset = None
        current_end_offset = None
        total_bytes = 0
        
        for file_offset, key, storage, nbytes in write_list:
            # Create byte tensor view of storage
            src_tensor = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
            
            # Check if we need to flush due to non-contiguous offset
            # Zipfile headers create ~128 byte gaps between storages
            if current_end_offset is not None:
                gap = file_offset - current_end_offset
                if gap > 512:  # Large gap - flush and start new batch
                    if self._io_buffer.get_offset() > 0:
                        self._flush_buffer(batch_start_offset)
                    batch_start_offset = file_offset
                    current_end_offset = file_offset
            else:
                batch_start_offset = file_offset
                current_end_offset = file_offset
            
            # Fill buffer with this storage's data
            src_offset = 0
            while src_offset < nbytes:
                fill_start = time.time()
                copied = self._io_buffer.fill(src_tensor, src_offset)
                self._stats['fill_buffer_secs'] += time.time() - fill_start
                self._stats['fill_buffer_count'] += 1
                
                src_offset += copied
                current_end_offset += copied
                
                # If buffer is full, drain it
                if self._io_buffer.is_full():
                    drain_bytes = self._io_buffer.get_offset()
                    
                    aio_start = time.time()
                    self._io_buffer.drain(drain_bytes, self._fd, batch_start_offset)
                    self._stats['aio_write_secs'] += time.time() - aio_start
                    self._stats['aio_bytes'] += drain_bytes
                    self._stats['write'] += 1
                    
                    batch_start_offset += drain_bytes
            
            total_bytes += nbytes
            self._stats['save_storage'] += 1
            self._stats['save_storage_bytes'] += nbytes
        
        # Flush any remaining data
        if self._io_buffer.get_offset() > 0:
            self._flush_buffer(batch_start_offset)
        
        # Wait for any pending async operations
        self._io_buffer.complete_ongoing_drain()
        
        self._stats['bytes'] = total_bytes
        self._stats['write_secs'] = time.time() - write_start
        
        return total_bytes
    
    def _flush_buffer(self, file_offset: int):
        """Flush the I/O buffer, handling aligned and unaligned portions."""
        aligned_bytes = self._io_buffer.get_aligned_num_bytes()
        unaligned_bytes = self._io_buffer.get_unaligned_num_bytes()
        
        if aligned_bytes > 0:
            aio_start = time.time()
            self._io_buffer.drain(aligned_bytes, self._fd, file_offset)
            self._stats['aio_write_secs'] += time.time() - aio_start
            self._stats['aio_bytes'] += aligned_bytes
            self._stats['write'] += 1
        
        # Wait for async drain to complete before handling unaligned
        self._io_buffer.complete_ongoing_drain()
        
        if unaligned_bytes > 0:
            # Handle unaligned tail with regular I/O
            slow_start = time.time()
            unaligned_tensor = torch.narrow(self._io_buffer.get_buffer(), 0, 0, unaligned_bytes)
            unaligned_offset = file_offset + aligned_bytes
            
            # Close O_DIRECT fd and use regular write for unaligned
            os.close(self._fd)
            with open(self._file_path, 'r+b') as f:
                f.seek(unaligned_offset)
                f.write(bytes(unaligned_tensor.cpu().numpy()))
            # Reopen fd
            self._fd = os.open(self._file_path, os.O_RDWR | os.O_DIRECT)
            
            self._stats['slow_write_secs'] += time.time() - slow_start
            self._stats['slow_bytes'] += unaligned_bytes
            self._stats['write'] += 1
        
        self._io_buffer.reset()
    
    def close(self):
        """Close the file descriptor."""
        try:
            os.close(self._fd)
        except:
            pass
        self._stats['close'] = 1
    
    def get_stats(self) -> dict:
        """Get statistics dictionary (matching FastFileWriter format)."""
        stats = self._stats.copy()
        if stats['write_secs'] > 0:
            stats['write_GB/s'] = stats['bytes'] / stats['write_secs'] / (1024**3)
        else:
            stats['write_GB/s'] = 0
        if stats['aio_write_secs'] > 0:
            stats['aio_GB/s'] = stats['aio_bytes'] / stats['aio_write_secs'] / (1024**3)
        else:
            stats['aio_GB/s'] = 0
        if stats['fill_buffer_secs'] > 0:
            stats['fill_buffer_GB/s'] = stats['aio_bytes'] / stats['fill_buffer_secs'] / (1024**3)
        else:
            stats['fill_buffer_GB/s'] = 0
        return stats


def _get_storage_offsets(file_path: str) -> Dict[str, int]:
    """Get file offsets for each storage record in a zip checkpoint."""
    reader = torch._C.PyTorchFileReader(file_path)
    offsets: Dict[str, int] = {}
    for record in reader.get_all_records():
        if record.startswith("data/"):
            key = record.split("/")[1]
            offsets[key] = reader.get_record_offset(record)
    return offsets


def _collect_serialized_storages_zip(obj: Any, pickle_protocol: int = 2) -> Dict[str, torch.UntypedStorage]:
    """
    Run a pickle pass with the same persistent_id logic as torch.serialization._save
    to collect storages and assign them keys.
    """
    from torch.serialization import normalize_storage_type, location_tag

    serialized_storages: Dict[str, torch.UntypedStorage] = {}
    id_map: Dict[int, str] = {}
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(o):
        if isinstance(o, torch.storage.TypedStorage) or torch.is_storage(o):
            if isinstance(o, torch.storage.TypedStorage):
                storage = o._untyped_storage
                storage_dtype = o.dtype
                storage_type_str = o._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = o._size()
            else:
                storage = o
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(o))
                storage_numel = storage.nbytes()

            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            if hasattr(o, "_fake_device") and o._fake_device is not None:
                loc = str(o._fake_device)
            else:
                loc = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, loc, storage_numel)

        return None

    buf = io.BytesIO()

    class _Pickler(pickle.Pickler):
        def persistent_id(self, o):
            return persistent_id(o)

    p = _Pickler(buf, protocol=pickle_protocol)
    p.dump(obj)

    return serialized_storages


def fastpersist_save(
    obj: Any,
    file_path: str,
    aio_handle,
    pinned_buffer: torch.Tensor,
    use_gds: bool = False,
    show_stats: bool = True,
    double_buffer: bool = True,
    pickle_protocol: int = 2,
) -> dict:
    """
    Save a PyTorch object using FastPersist without patching torch,
    using the non-legacy zipfile format.
    
    This function:
    1. Uses torch.serialization.skip_data() to create a zipfile skeleton
    2. Gets storage offsets using PyTorchFileReader.get_record_offset()
    3. Uses FastPersistScatterWriter (FastPersist I/O) to write storages at offsets
    
    Args:
        obj: The PyTorch object to save (model, state_dict, etc.)
        file_path: Path to save the checkpoint
        aio_handle: DeepSpeed AIO or GDS handle
        pinned_buffer: Pinned memory buffer for async I/O
        use_gds: Whether to use GPU Direct Storage
        show_stats: Whether to print statistics
        double_buffer: Whether to use double buffering
        pickle_protocol: Pickle protocol version
    
    Returns:
        Statistics dictionary matching FastFileWriter format
    """
    total_start = time.time()
    
    # Step 1: Write skeleton using skip_data + non-legacy zip format
    skeleton_start = time.time()
    with torch.serialization.skip_data():
        torch.save(obj, file_path, _use_new_zipfile_serialization=True)
    skeleton_secs = time.time() - skeleton_start
    
    # Step 2: Get storage offsets from the zipfile
    offsets = _get_storage_offsets(file_path)
    
    # Step 3: Collect storages with matching keys
    serialized_storages = _collect_serialized_storages_zip(obj, pickle_protocol)
    
    # Sanity check
    if set(offsets.keys()) != set(serialized_storages.keys()):
        raise RuntimeError(
            f"Mismatch between skeleton storage keys and collected storages"
        )
    
    # Step 4: Create FastPersist scatter writer and write storages
    config = FastFileWriterConfig(
        dnvme_handle=aio_handle,
        pinned_tensor=pinned_buffer,
        double_buffer=double_buffer,
    )
    
    writer = FastPersistScatterWriter(file_path, config)
    try:
        writer.save_storages_at_offsets(serialized_storages, offsets, use_gds)
    finally:
        writer.close()
    
    # Get stats and add skeleton time
    stats = writer.get_stats()
    stats['skeleton_secs'] = skeleton_secs
    
    if show_stats:
        print(f"stats = {stats}")
    
    return stats


# Convenience functions for creating handles and buffers

def get_aio_handle(
    block_size: int = 16 * (1024**2),
    queue_depth: int = 32,
    single_submit: bool = False,
    overlap_events: bool = True,
    intra_op_parallelism: int = 16,
):
    """Create an async I/O handle for CPU-based NVMe operations."""
    return AsyncIOBuilder().load(verbose=False).aio_handle(
        block_size=block_size,
        queue_depth=queue_depth,
        single_submit=single_submit,
        overlap_events=overlap_events,
        intra_op_parallelism=intra_op_parallelism,
    )


def get_gds_handle(
    block_size: int = 16 * (1024**2),
    queue_depth: int = 32,
    single_submit: bool = False,
    overlap_events: bool = True,
    intra_op_parallelism: int = 16,
):
    """Create a GDS handle for GPU Direct Storage operations."""
    return GDSBuilder().load(verbose=False).gds_handle(
        block_size=block_size,
        queue_depth=queue_depth,
        single_submit=single_submit,
        overlap_events=overlap_events,
        intra_op_parallelism=intra_op_parallelism,
    )


def get_pinned_buffer(size_mb: int = 64, use_gds: bool = False, gds_handle=None):
    """Create a pinned memory buffer for async I/O."""
    size_bytes = size_mb * (1024**2)

    if use_gds:
        buffer = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=get_accelerator().current_device_name(),
        )
        if gds_handle is None:
            gds_handle = get_gds_handle()
        gds_handle.pin_device_tensor(buffer)
        return buffer
    else:
        return torch.zeros(size_bytes, dtype=torch.uint8, device="cpu").pin_memory()
