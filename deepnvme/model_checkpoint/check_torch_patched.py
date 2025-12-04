#!/usr/bin/env python3
"""
Check if torch.serialization is patched for FastPersist.

Patched torch includes modifications to use FastFileWriter in _legacy_save.
"""

import sys


def check_torch_patched():
    """
    Returns True if torch.serialization appears to be patched for FastPersist.
    """
    try:
        import torch
        import inspect
        from torch import serialization
    except ImportError:
        print("ERROR: torch not installed")
        return None

    # Get the source of _legacy_save if available
    try:
        source = inspect.getsource(serialization._legacy_save)
    except (TypeError, OSError):
        print("WARNING: Could not inspect _legacy_save source")
        source = ""

    # Check for FastPersist indicators in _legacy_save
    fastpersist_indicators = [
        "FastFileWriter",
        "save_torch_storage_object_list",
        "dnvme",
        "_should_use_fast_file_writer",
    ]

    found_indicators = [ind for ind in fastpersist_indicators if ind in source]

    # Also check if the module has FastPersist-related attributes
    module_indicators = []
    if hasattr(serialization, "_should_use_fast_file_writer"):
        module_indicators.append("_should_use_fast_file_writer")
    if hasattr(serialization, "FastFileWriter"):
        module_indicators.append("FastFileWriter")

    is_patched = len(found_indicators) > 0 or len(module_indicators) > 0

    return {
        "is_patched": is_patched,
        "torch_version": torch.__version__,
        "serialization_file": serialization.__file__,
        "found_in_source": found_indicators,
        "found_in_module": module_indicators,
    }


def main():
    print("=" * 60)
    print("Torch Serialization Patch Check")
    print("=" * 60)

    result = check_torch_patched()

    if result is None:
        sys.exit(1)

    print(f"Torch version:      {result['torch_version']}")
    print(f"Serialization file: {result['serialization_file']}")
    print()

    if result["is_patched"]:
        print("STATUS: PATCHED ")
    else:
        print("STATUS: NOT PATCHED")

    print("=" * 60)
    return 0 if not result["is_patched"] else 0


if __name__ == "__main__":
    sys.exit(main())

