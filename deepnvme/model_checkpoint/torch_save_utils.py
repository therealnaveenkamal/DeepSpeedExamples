import time
import torch
import os
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder, GDSBuilder
from deepspeed.io import MockFileWriter, PyFileWriter, FastFileWriter, FastFileWriterConfig
from deepspeed.accelerator import get_accelerator

AIO_QUEUE_DEPTH = 8
AIO_BLOCK_SIZE = 8 * (1024**2)
AIO_INTRA_OP_PARALLEL = 1
AIO_SINGLE_SUBMIT = False
AIO_OVERLAP_EVENTS = False
PINNED_BUFFER_MB = 64

def load_io_ops(args):
    if AsyncIOBuilder().is_compatible(): 
        AsyncIOBuilder().load(verbose=False)
    if args.gpu and GDSBuilder().is_compatible():
        GDSBuilder().load(verbose=False)


def _get_aio_handle():
    h = AsyncIOBuilder().load(verbose=False).aio_handle(block_size=AIO_BLOCK_SIZE,
                                           queue_depth=AIO_QUEUE_DEPTH,
                                           single_submit=AIO_SINGLE_SUBMIT,
                                           overlap_events=AIO_SINGLE_SUBMIT,
                                           intra_op_parallelism=AIO_INTRA_OP_PARALLEL)
    return h

def _get_gds_handle():
    h = GDSBuilder().load(verbose=False).gds_handle(block_size=AIO_BLOCK_SIZE,
                                    queue_depth=AIO_QUEUE_DEPTH,
                                    single_submit=AIO_SINGLE_SUBMIT,
                                    overlap_events=AIO_SINGLE_SUBMIT,
                                    intra_op_parallelism=AIO_INTRA_OP_PARALLEL)
    return h

def test_save(file, buffer, args):
    st = time.time()
    torch.save(f=file,
               obj=buffer,
               _use_new_zipfile_serialization=args.zipfile)
    return time.time() - st


def test_ds_mock_save(file, buffer, args):
    st = time.time()
    ds_mock_writer = MockFileWriter(file)
    torch.save(f=ds_mock_writer,
               obj=buffer,
               _use_new_zipfile_serialization=args.zipfile)
    ds_mock_writer.close()  # Force flush to storage
    write_sec = time.time() - st
    if not args.no_statistics:
        ds_mock_writer._dump_state()
    return write_sec


def test_ds_py_save(file, buffer, args):
    st = time.time()
    ds_py_writer = PyFileWriter(file)
    torch.save(f=ds_py_writer,
               obj=buffer,
               _use_new_zipfile_serialization=args.zipfile)
    ds_py_writer.close()  # Force flush to storage
    write_sec = time.time() - st
    if not args.no_statistics:
        ds_py_writer._dump_state()
    return write_sec

def _get_aio_components(args):
    h = _get_aio_handle()
    pinned_memory = torch.zeros(args.io_buffer_mb * (1024**2),
                                dtype=torch.uint8,
                                device='cpu').pin_memory()
    return h, pinned_memory

def _get_gds_components(args):
    h = _get_gds_handle()
    pinned_memory = torch.empty(args.io_buffer_mb * (1024**2), 
                                dtype=torch.uint8, 
                                device=get_accelerator().current_device_name())
    h.pin_device_tensor(pinned_memory)
    return h, pinned_memory



def _test_ds_fast_save(file, buffer, args, use_gds):
    if use_gds:
        h, pinned_memory = _get_gds_components(args)
    else:
        h, pinned_memory = _get_aio_components(args)
    st = time.time()
    fast_writer_config = FastFileWriterConfig(dnvme_handle=h,
                                  pinned_tensor=pinned_memory,
                                  double_buffer=not args.single_io_buffer,
                                  num_parallel_writers=1,
                                  writer_rank=0)

    ds_fast_writer = FastFileWriter(file_path=file,
                                    config=fast_writer_config)
    torch.save(f=ds_fast_writer,
               obj=buffer,
               _use_new_zipfile_serialization=args.zipfile)
    ds_fast_writer.close()  # Force flush to storage
    write_sec = time.time() - st
    if not args.no_statistics:
        ds_fast_writer._dump_state()
    return write_sec


def test_ds_aio_fast_save(file, buffer, args):
    return _test_ds_fast_save(file, buffer, args, False)

def test_ds_gds_fast_save(file, buffer, args):
    return _test_ds_fast_save(file, buffer, args, True)

def _get_storage_map(real_obj, fake_obj, storage_map):
    if isinstance(real_obj, torch.Tensor):
        if hasattr(fake_obj.untyped_storage(), '_checkpoint_offset'):
            offset = fake_obj.untyped_storage()._checkpoint_offset
            storage_map[offset] = real_obj.untyped_storage()
    elif isinstance(real_obj, (list, tuple)):
        for r, f in zip(real_obj, fake_obj):
            _get_storage_map(r, f, storage_map)
    elif isinstance(real_obj, dict):
        for k in real_obj:
            _get_storage_map(real_obj[k], fake_obj[k], storage_map)
    elif hasattr(real_obj, 'state_dict'):
        # For modules/optimizers, traverse their state_dict to find tensors
        _get_storage_map(real_obj.state_dict(), fake_obj.state_dict(), storage_map)

def test_ds_aio_no_patch_save(file, buffer, args):
    st = time.time()
    
    with torch.serialization.skip_data():
        torch.save(buffer, file, _use_new_zipfile_serialization=True)
    
    from torch._subclasses import FakeTensorMode
    with FakeTensorMode():
        fake_buffer = torch.load(file, weights_only=False)
    storage_map = {}
    _get_storage_map(buffer, fake_buffer, storage_map)
    
    # Sort by offset to write sequentially where possible (good for performance)
    sorted_offsets = sorted(storage_map.keys())
    # Tag real storages with offsets so FastFileWriter knows where to put them
    sorted_storages = []
    for offset in sorted_offsets:
        storage = storage_map[offset]
        storage._checkpoint_offset = offset
        # FastFileWriter expects (storage, dtype) tuple for new torch versions
        sorted_storages.append((storage, torch.uint8))
        
    # 4. Write data using FastFileWriter
    # Setup writer
    use_gds = False # Default to AIO for now, or check args?
    if use_gds:
        h, pinned_memory = _get_gds_components(args)
    else:
        h, pinned_memory = _get_aio_components(args)
        
    fast_writer_config = FastFileWriterConfig(dnvme_handle=h,
                                  pinned_tensor=pinned_memory,
                                  double_buffer=not args.single_io_buffer,
                                  num_parallel_writers=1,
                                  writer_rank=0)

    ds_fast_writer = FastFileWriter(file_path=file,
                                    config=fast_writer_config)
    
    # Write the storages. save_size=False because zipfile format doesn't prefix data with size in the blob.
    ds_fast_writer.save_torch_storage_object_list(sorted_storages, save_size=False)
    
    ds_fast_writer.close()
    
    write_sec = time.time() - st
    if not args.no_statistics:
        ds_fast_writer._dump_state()
    return write_sec
