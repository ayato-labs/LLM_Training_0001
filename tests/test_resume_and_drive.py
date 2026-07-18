import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.training.model_utils as utils

def run_tests():
    print("==========================================================")
    print("Running Standalone Diagnostics for Resume & Checkpoint Cleanup")
    print("==========================================================")

    # Setup temporary directory for OUTPUT_DIR
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Test 1: cleanup_old_checkpoints
        print("\nTest 1: cleanup_old_checkpoints behavior check...")
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
        
        # Test 1a: Cleanup keeping 2 checkpoints
        utils.cleanup_old_checkpoints(keep=2, output_dir=temp_dir)
        print(f"  Exists check (keep=2): cp100={cp100.exists()} (should be False), cp200={cp200.exists()}, cp300={cp300.exists()}")
        
        # Test 2: Resume latest checkpoint path resolution
        print("\nTest 2: checkpoint-latest resolution check...")
        checkpoints = utils.get_checkpoints(output_dir=temp_dir)
        print("  Found checkpoints:", checkpoints)
        if checkpoints:
            latest = checkpoints[-1][1]
            print("  Latest checkpoint path resolved to:", latest)
            assert latest == cp300, f"Expected {cp300}, but got {latest}"
        else:
            print("  [BUG CONFIRMED] No local checkpoints found.")

        # Test 3: Chronological (mtime) checkpoint-latest resolution check
        print("\nTest 3: checkpoint-latest chronological resolution check...")
        # Modify cp200 to simulate a newer run checkpoint
        time_touched = (cp200 / "trainer_state.json")
        # Ensure it has a distinct newer modification time
        import time
        os.utime(time_touched, (time.time() + 10, time.time() + 10))
        checkpoints_mtime = utils.get_checkpoints(output_dir=temp_dir, sort_by="mtime")
        print("  Found checkpoints (mtime):", checkpoints_mtime)
        if checkpoints_mtime:
            latest_mtime = checkpoints_mtime[-1][1]
            print("  Latest checkpoint path (mtime) resolved to:", latest_mtime)
            assert latest_mtime == cp200, f"Expected {cp200} because it was modified last, but got {latest_mtime}"
        else:
            print("  [BUG] No local checkpoints found for mtime test.")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    print("\n==========================================================")
    print("Diagnostics Completed.")
    print("==========================================================")

if __name__ == "__main__":
    run_tests()
