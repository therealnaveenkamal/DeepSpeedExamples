import time
import torch
import os
import deepspeed
from deepspeed.ops.op_builder import AsyncIOBuilder, GDSBuilder
from deepspeed.io import MockFileWriter, PyFileWriter, FastFileWriter, FastFileWriterConfig
from deepspeed.accelerator import get_accelerator
from fastpersist_save import fastpersist_save, fastpersist_save_legacy

# Constants for FastPersist I/O
AIO_QUEUE_DEPTH = 64
AIO_BLOCK_SIZE = 16 * (1024**2)
AIO_INTRA_OP_PARALLEL = 4
AIO_SINGLE_SUBMIT = False
AIO_OVERLAP_EVENTS = True
PINNED_BUFFER_MB = 256

def load_io_ops(args):
    if AsyncIOBuilder().is_compatible(): 
        AsyncIOBuilder().load(verbose=False)
    if args.gpu and GDSBuilder().is_compatible():
        GDSBuilder().load(verbose=False)


def _get_aio_handle():
    h = AsyncIOBuilder().load(verbose=False).aio_handle(block_size=AIO_BLOCK_SIZE,
                                           queue_depth=AIO_QUEUE_DEPTH,
                                           single_submit=AIO_SINGLE_SUBMIT,
                                           overlap_events=AIO_OVERLAP_EVENTS,
                                           intra_op_parallelism=AIO_INTRA_OP_PARALLEL)
    return h

def _get_gds_handle():
    h = GDSBuilder().load(verbose=False).gds_handle(block_size=AIO_BLOCK_SIZE,
                                    queue_depth=AIO_QUEUE_DEPTH,
                                    single_submit=AIO_SINGLE_SUBMIT,
                                    overlap_events=AIO_OVERLAP_EVENTS,
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
                                  num_parallel_writers=8,
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


def _test_fastpersist_nopatch(file, buffer, args, use_gds):
    """
    Test FastPersist using no-patch approach.
    
    If args.zipfile is True:
      Uses FastFileWriter wrapper (sequential O_DIRECT). 
      Limited by Python zip overhead (~1.2 GB/s).
      
    If args.zipfile is False (legacy):
      Uses manual legacy serialization logic (fastpersist_save_legacy).
      Hits maximum FastPersist speed (~10+ GB/s).
    """
    if use_gds:
        h, pinned_memory = _get_gds_components(args)
    else:
        h, pinned_memory = _get_aio_components(args)
    
    st = time.time()
    
    if args.zipfile:
        # Best we can do for zipfile without patching
        stats = fastpersist_save(
            obj=buffer,
            file_path=file,
            aio_handle=h,
            pinned_buffer=pinned_memory,
            use_gds=use_gds,
            show_stats=not args.no_statistics,
            double_buffer=not args.single_io_buffer,
            num_parallel_writers=8
        )
    else:
        # Fast legacy path (manual serialization)
        stats = fastpersist_save_legacy(
            obj=buffer,
            file_path=file,
            aio_handle=h,
            pinned_buffer=pinned_memory,
            use_gds=use_gds,
            show_stats=not args.no_statistics,
            double_buffer=not args.single_io_buffer,
            num_parallel_writers=8
        )
        
    write_sec = time.time() - st
    
    return write_sec


def test_fastpersist_aio_nopatch(file, buffer, args):
    """FastPersist with async I/O (no torch patching required)."""
    return _test_fastpersist_nopatch(file, buffer, args, False)


def test_fastpersist_gds_nopatch(file, buffer, args):
    """FastPersist with GPU Direct Storage (no torch patching required)."""
    return _test_fastpersist_nopatch(file, buffer, args, True)
