# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
FastPersist save implementation without patching torch, using the
**non-legacy zipfile path**.

This module:
1. Uses `torch.serialization.skip_data()` to write a zipfile skeleton
   (metadata + reserved regions for storages) via the standard
   non-legacy `torch.save` path.
2. Uses `PyTorchFileReader.get_record_offset()` to get the file offsets
   for each storage record.
3. Re-runs the same persistent_id logic used by torch's `_save` to
   collect the actual storages and associate them with the same keys.
4. Uses DeepSpeed's async I/O handle (`aio_handle.pwrite`) to scatter-
   write storage bytes directly into the reserved regions.

No modifications to `torch/serialization.py` are required, and the
resulting checkpoint is compatible with `torch.load()`.
"""

import io
import os
import time
import pickle
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from deepspeed.ops.op_builder import AsyncIOBuilder, GDSBuilder
from deepspeed.accelerator import get_accelerator


@dataclass
class FastPersistStats:
    """Statistics for FastPersist save operation."""
    total_time: float = 0.0
    skeleton_time: float = 0.0
    storage_write_time: float = 0.0
    num_storages: int = 0
    total_bytes: int = 0

    def __str__(self):
        speed = self.total_bytes / self.total_time / (1024**3) if self.total_time > 0 else 0
        return (
            f"FastPersist: {self.total_bytes / (1024**3):.2f} GB, "
            f"{self.total_time:.2f} secs, {speed:.2f} GB/s "
            f"(skeleton: {self.skeleton_time:.3f}s, write: {self.storage_write_time:.3f}s, "
            f"storages: {self.num_storages})"
        )


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
        # This is adapted from torch.serialization._save
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
            # We only care about the storage object; dtype/location are not needed here
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, loc, storage_numel)

        return None

    # Run a pickle pass purely to populate serialized_storages / id_map
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
    pickle_protocol: int = 2,
) -> FastPersistStats:
    """
    Save a PyTorch object using FastPersist without patching torch,
    using the non-legacy zipfile format.

    Steps:
      1. Use `torch.serialization.skip_data()` + `torch.save` to write
         the zip skeleton (metadata + reserved storage regions).
      2. Use `PyTorchFileReader` to get the file offset of each storage.
      3. Re-run persistent_id logic to get the storages and their keys.
      4. Use `aio_handle.pwrite` to scatter-write storage bytes directly,
         batching writes through the pinned buffer for efficiency.
    """
    stats = FastPersistStats()
    total_start = time.time()

    # Step 1: write skeleton using skip_data + non-legacy zip format
    skeleton_start = time.time()
    with torch.serialization.skip_data():
        # Force new zipfile serialization to avoid legacy path
        torch.save(obj, file_path, _use_new_zipfile_serialization=True)
    stats.skeleton_time = time.time() - skeleton_start

    # Step 2: get storage offsets from the zipfile
    offsets = _get_storage_offsets(file_path)

    # Step 3: collect storages with matching keys
    serialized_storages = _collect_serialized_storages_zip(obj, pickle_protocol)
    stats.num_storages = len(serialized_storages)

    # Sanity check: we should have matching key sets
    if set(offsets.keys()) != set(serialized_storages.keys()):
        raise RuntimeError(
            f"Mismatch between skeleton storage keys and collected storages:\n"
            f"  offsets keys: {sorted(offsets.keys())[:5]} ...\n"
            f"  storages keys: {sorted(serialized_storages.keys())[:5]} ..."
        )

    # Step 4: scatter-write storages using batched async pwrite
    write_start = time.time()
    total_bytes = _batched_scatter_write(
        file_path, serialized_storages, offsets, aio_handle, pinned_buffer, use_gds
    )

    stats.storage_write_time = time.time() - write_start
    stats.total_bytes = total_bytes
    stats.total_time = time.time() - total_start

    if show_stats:
        print(stats)

    return stats


def _batched_scatter_write(
    file_path: str,
    serialized_storages: Dict[str, torch.UntypedStorage],
    offsets: Dict[str, int],
    aio_handle,
    pinned_buffer: torch.Tensor,
    use_gds: bool = False,
) -> int:
    """
    Write storages to file using batched I/O through pinned buffer.
    
    Since zip file headers create only 128-byte gaps between storages,
    we can batch multiple small storages into single large writes for
    much better I/O efficiency.
    
    Large storages (> buffer_size) are written directly without buffering.
    """
    buffer_size = pinned_buffer.numel()
    total_bytes = 0
    
    # Build sorted list of (offset, key, storage, nbytes)
    write_list = []
    for key, storage in serialized_storages.items():
        offset = offsets[key]
        nbytes = storage.nbytes()
        
        # Move to appropriate device
        if use_gds:
            if storage.device.type != get_accelerator().device_name():
                storage = storage.to(get_accelerator().current_device_name())
        else:
            if storage.device.type != "cpu":
                storage = storage.cpu()
        
        write_list.append((offset, key, storage, nbytes))
    
    # Sort by file offset for sequential access pattern
    write_list.sort(key=lambda x: x[0])
    
    # Batch writes: accumulate in pinned buffer, flush when full or gap is too large
    buffer_offset = 0
    batch_start_offset = None
    pending_copies = []  # List of (buffer_pos, storage, nbytes)
    
    GAP_THRESHOLD = 512  # Max gap to bridge (zip headers are 128 bytes)
    
    for file_offset, key, storage, nbytes in write_list:
        # Handle large storages directly (larger than buffer)
        if nbytes > buffer_size:
            # Flush any pending batch first
            if pending_copies:
                _flush_batch(aio_handle, pinned_buffer, file_path,
                            batch_start_offset, buffer_offset, pending_copies)
                buffer_offset = 0
                pending_copies = []
                batch_start_offset = None
            
            # Write large storage directly in chunks
            _write_large_storage(aio_handle, pinned_buffer, file_path,
                                file_offset, storage, nbytes)
            total_bytes += nbytes
            continue
        
        # Check if we should start a new batch
        start_new_batch = False
        
        if batch_start_offset is None:
            # First storage
            start_new_batch = True
        else:
            # Check gap from end of current batch to this storage
            batch_end = batch_start_offset + buffer_offset
            gap = file_offset - batch_end
            
            if gap > GAP_THRESHOLD:
                # Gap too large, flush current batch and start new one
                start_new_batch = True
            elif buffer_offset + gap + nbytes > buffer_size:
                # Won't fit in buffer, flush and start new
                start_new_batch = True
        
        if start_new_batch and buffer_offset > 0:
            # Flush current batch
            _flush_batch(aio_handle, pinned_buffer, file_path, 
                        batch_start_offset, buffer_offset, pending_copies)
            buffer_offset = 0
            pending_copies = []
            batch_start_offset = None
        
        if batch_start_offset is None:
            batch_start_offset = file_offset
        
        # Calculate position in buffer (accounting for gaps)
        buffer_pos = file_offset - batch_start_offset
        
        # If there's a gap, we need to skip over it in the buffer
        # (the gap contains zip headers which are already written)
        if buffer_pos > buffer_offset:
            # Small gap - just advance buffer_offset (headers already in file)
            buffer_offset = buffer_pos
        
        # Queue this storage for copying
        pending_copies.append((buffer_pos, storage, nbytes))
        buffer_offset = buffer_pos + nbytes
        total_bytes += nbytes
    
    # Flush final batch
    if pending_copies:
        _flush_batch(aio_handle, pinned_buffer, file_path,
                    batch_start_offset, buffer_offset, pending_copies)
    
    # Wait for all async writes
    aio_handle.wait()
    
    return total_bytes


def _write_large_storage(
    aio_handle,
    pinned_buffer: torch.Tensor,
    file_path: str,
    file_offset: int,
    storage: torch.UntypedStorage,
    nbytes: int,
):
    """Write a large storage in chunks through the pinned buffer."""
    buffer_size = pinned_buffer.numel()
    src = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
    
    src_offset = 0
    current_file_offset = file_offset
    
    while src_offset < nbytes:
        chunk_size = min(buffer_size, nbytes - src_offset)
        
        # Copy chunk to pinned buffer
        pinned_buffer[:chunk_size].copy_(src[src_offset:src_offset + chunk_size])
        
        # Write chunk
        aio_handle.pwrite(pinned_buffer[:chunk_size], file_path, False, True, current_file_offset)
        
        src_offset += chunk_size
        current_file_offset += chunk_size


def _flush_batch(
    aio_handle,
    pinned_buffer: torch.Tensor,
    file_path: str,
    batch_start_offset: int,
    batch_size: int,
    pending_copies: List[Tuple[int, torch.UntypedStorage, int]],
):
    """Copy storages to pinned buffer and issue async write."""
    # Copy each storage to its position in the buffer
    for buffer_pos, storage, nbytes in pending_copies:
        # Create byte tensor view of storage
        src = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
        # Copy to pinned buffer
        pinned_buffer[buffer_pos:buffer_pos + nbytes].copy_(src)
    
    # Issue async write for the entire batch
    write_tensor = pinned_buffer[:batch_size]
    aio_handle.pwrite(write_tensor, file_path, False, True, batch_start_offset)


# Convenience functions for getting handles and buffers

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
    """Create a pinned memory buffer for async I/O (kept for API compatibility)."""
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
