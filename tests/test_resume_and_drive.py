import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.drive_uploader as uploader

def run_tests():
    print("==========================================================")
    print("Running Standalone Diagnostics for Resume & Drive Cleanup")
    print("==========================================================")

    # Setup temporary directory for OUTPUT_DIR
    temp_dir = Path(tempfile.mkdtemp())
    old_output_dir = uploader.OUTPUT_DIR
    uploader.OUTPUT_DIR = temp_dir

    try:
        # Test 1: get_service unpacking failure
        print("\nTest 1: DriveUploadCallback._get_service unpacking check...")
        callback = uploader.DriveUploadCallback()
        uploader.HAS_GOOGLE_DRIVE = True
        
        try:
            # Under buggy behavior, this will fail with TypeError
            res = callback._get_service()
            print("  Returned value:", res)
            if res is None:
                # Triggers unpacking error when calling `service, folder_id = self._get_service()`
                print("  [BUG CONFIRMED] Returned None instead of (None, None).")
                print("  This triggers: 'TypeError: cannot unpack non-iterable NoneType object'")
            else:
                service, folder_id = res
                print("  Successfully unpacked service and folder_id.")
        except TypeError as e:
            print("  [BUG CONFIRMED] TypeError raised during unpacking: ", e)

        # Test 2: cleanup_old_checkpoints
        print("\nTest 2: cleanup_old_checkpoints behavior check...")
        cp100 = temp_dir / "checkpoint-100"
        cp200 = temp_dir / "checkpoint-200"
        cp300 = temp_dir / "checkpoint-300"
        
        cp100.mkdir()
        cp200.mkdir()
        cp300.mkdir()
        
        (cp100 / "trainer_state.json").touch()
        (cp200 / "trainer_state.json").touch()
        (cp300 / "trainer_state.json").touch()

        print("  Created cp100, cp200, cp300.")
        
        # Test 2a: No checkpoints are uploaded
        uploader.cleanup_old_checkpoints(keep=2)
        print(f"  Without .uploaded flag, exists check: cp100={cp100.exists()}, cp200={cp200.exists()}, cp300={cp300.exists()}")
        
        # Test 2b: Mark cp100 as uploaded
        (cp100 / ".uploaded").touch()
        uploader.cleanup_old_checkpoints(keep=2)
        print(f"  With cp100 uploaded, exists check: cp100={cp100.exists()} (should be False), cp200={cp200.exists()}, cp300={cp300.exists()}")

        # Test 3: Resume latest checkpoint path resolution
        print("\nTest 3: checkpoint-latest resolution check...")
        from src.training.drive_uploader import get_checkpoints
        checkpoints = get_checkpoints()
        print("  Found checkpoints:", checkpoints)
        if checkpoints:
            latest = checkpoints[-1][1]
            print("  Latest checkpoint path resolved to:", latest)
        else:
            print("  [BUG CONFIRMED] No local checkpoints found (if all deleted or no matching pattern).")

    finally:
        # Restore output dir
        uploader.OUTPUT_DIR = old_output_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    print("\n==========================================================")
    print("Diagnostics Completed.")
    print("==========================================================")

if __name__ == "__main__":
    run_tests()
