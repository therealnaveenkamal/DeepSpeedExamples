[FastPersist](https://arxiv.org/abs/2406.13768) is an optimization technique that leverages NVMe storage to accelerate model checkpointing. This folder contains micro-benchmarks and instructions for demonstrating FastPersist. 

## Enabling FastPersist Optimizations ##

FastPersist is designed to integrate with torch checkpointing and has been validated with torch version 2.6.0.

### Option 1: No-Patch Approach (Recommended) ###

Starting with PyTorch 2.6, FastPersist can be used **without modifying torch's source code** by leveraging `torch.serialization.skip_data()`. This approach:

1. Uses `torch.serialization.skip_data()` to save the checkpoint skeleton (metadata + reserved storage regions) in the standard zipfile format
2. Extracts storage offsets using `PyTorchFileReader.get_record_offset()`
3. Batches multiple storages together and writes them directly to the correct file offsets using DeepSpeed's async NVMe I/O (`aio_handle.pwrite`)

The no-patch implementation is in `fastpersist_save.py` and is automatically used by the benchmark scripts via the `test_fastpersist_aio_nopatch` and `test_fastpersist_gds_nopatch` test functions.

**Performance**: The no-patch approach achieves ~70-80% of the patched approach's throughput while maintaining full compatibility with standard `torch.load()`.

### Option 2: Patched Torch Approach (Legacy) ###

For maximum performance or legacy use, we also provide patched versions of torch's serialization.py that integrate FastPersist directly into `torch.save()`. See [original](torch/serialization_orig_v2.6.0.py) and [patched](torch/serialization_fast_v2.6.0.py) versions. To use this approach, overwrite `torch/serialization.py` in your torch installation with the patched version.

The patched approach uses the legacy serialization format which writes storages sequentially, enabling optimal NVMe throughput.

## Available Micro-benchmarks ##
This folder contains three different micro-benchmarks that are implemented by the following scripts:
1. torch_save_tensor.py: Serialize a raw pytorch tensor to disk using `torch.save()` integration.
2. torch_save_model.py: Serialize a HF model to disk using `torch.save()` integration. 
3. deepspeed_save_model.py: Serialize a HF model to disk using DeepSpeed integration. 

Each script provides a `--help` option to examine the available configurations. The scripts are written for single-process execution and so can be launched using `python`. 

As an example, the performance of using the `torch.save()` integration of checkpointing HF phi-3-mini model can be measured as follows: 
```bash
python torch_save_model.py --model phi3 --folder /mnt/nvme0 --io_buffer_mb 256
```

The script executes and reports the performance of the checkpointing workload using different mechanisms including:
- `test_save`: Vanilla `torch.save()`
- `test_ds_aio_fast_save`: FastPersist with CPU bounce buffer (requires torch patching)
- `test_fastpersist_aio_nopatch`: FastPersist with CPU bounce buffer (no patching required)
- `test_fastpersist_gds_nopatch`: FastPersist with GPU Direct Storage (no patching required, requires `--gpu`)

Example results on a single NVMe SSD:

```bash
test_save                      -- 14.23 GB, 11.40 secs,  1.25 GB/s
test_ds_aio_fast_save          -- 14.23 GB,  1.30 secs, 10.93 GB/s  (patched)
test_fastpersist_aio_nopatch   -- 14.23 GB,  1.61 secs,  8.83 GB/s  (no-patch)
```

For best performance with the no-patch approach, use `--io_buffer_mb 256` (or tune based on your system).

## API Usage ##

For programmatic use of FastPersist without patching torch:

```python
from fastpersist_save import fastpersist_save, get_aio_handle, get_pinned_buffer

# Get async I/O handle and pinned buffer (256MB recommended)
aio_handle = get_aio_handle()
pinned_buffer = get_pinned_buffer(size_mb=256)

# Save model checkpoint using FastPersist
model_state = model.state_dict()
stats = fastpersist_save(
    obj=model_state,
    file_path="/path/to/checkpoint.pt",
    aio_handle=aio_handle,
    pinned_buffer=pinned_buffer
)

# The checkpoint can be loaded with standard torch.load()
loaded_state = torch.load("/path/to/checkpoint.pt", weights_only=True)
```
