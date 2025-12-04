[FastPersist](https://arxiv.org/abs/2406.13768) is an optimization technique that leverages NVMe storage to accelerate model checkpointing. This folder contains micro-benchmarks and instructions for demonstrating FastPersist. 

## Enabling FastPersist Optimizations ##

FastPersist is designed to integrate with torch checkpointing and has been validated with torch version 2.6.0.

### Option 1: No-Patch Approach (Recommended) ###

FastPersist can be used **without modifying torch's source code** by using `FastFileWriter` wrappers.

1. **Standard (Zipfile) Format**: Use `fastpersist_save()`. This wraps `torch.save` and passes `FastFileWriter` as the file object, enabling sequential async I/O. Performance is ~1.2 GB/s (limited by zip overhead).
2. **Legacy Format (Fastest)**: Use `fastpersist_save_legacy()`. This manually serializes the object in the PyTorch legacy format, leveraging `FastFileWriter`'s batch API for maximum throughput. **Performance: ~10+ GB/s**.

The implementation is in `fastpersist_save.py`.

### Option 2: Patched Torch Approach (Legacy) ###

For systems where patching is preferred, we provide patched versions of torch's serialization.py. See [original](torch/serialization_orig_v2.6.0.py) and [patched](torch/serialization_fast_v2.6.0.py) versions.

## Available Micro-benchmarks ##
This folder contains three different micro-benchmarks that are implemented by the following scripts:
1. torch_save_tensor.py: Serialize a raw pytorch tensor to disk using `torch.save()` integration.
2. torch_save_model.py: Serialize a HF model to disk using `torch.save()` integration. 
3. deepspeed_save_model.py: Serialize a HF model to disk using DeepSpeed integration. 

Each script provides a `--help` option to examine the available configurations.

### Benchmark Usage

To measure performance of checkpointing HF phi-3-mini model:
```bash
python torch_save_model.py --model phi3 --folder /mnt/nvme0 --io_buffer_mb 256
```

Results comparison:
- `test_save`: Vanilla `torch.save()` (~1.2 GB/s)
- `test_ds_aio_fast_save`: FastPersist using FastFileWriter (~1.5 GB/s)
- `test_fastpersist_aio_nopatch`: **FastPersist Optimized Legacy No-Patch (~13 GB/s)**

To test the zipfile format (slower due to format overhead):
```bash
python torch_save_model.py --model phi3 --folder /mnt/nvme0 --io_buffer_mb 256 --zipfile
```

## API Usage ##

For programmatic use of FastPersist without patching torch:

```python
from fastpersist_save import fastpersist_save, fastpersist_save_legacy, get_aio_handle, get_pinned_buffer

# Get async I/O handle and pinned buffer (256MB recommended)
aio_handle = get_aio_handle()
pinned_buffer = get_pinned_buffer(size_mb=256)

# OPTION 1: Maximum Speed (Legacy Format) - ~10+ GB/s
stats = fastpersist_save_legacy(
    obj=model.state_dict(),
    file_path="/path/to/checkpoint_legacy.pt",
    aio_handle=aio_handle,
    pinned_buffer=pinned_buffer
)

# OPTION 2: Standard Format (Zipfile) - ~1.2 GB/s
stats = fastpersist_save(
    obj=model.state_dict(),
    file_path="/path/to/checkpoint_zip.pt",
    aio_handle=aio_handle,
    pinned_buffer=pinned_buffer
)

# Both checkpoints can be loaded with standard torch.load()
loaded = torch.load("/path/to/checkpoint.pt")
```
