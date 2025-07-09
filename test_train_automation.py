import sys
import subprocess
from pathlib import Path

def find_best_checkpoint(checkpoints_root: Path, max_r: float, max_t: float) -> Path:
    """Locate the checkpoint file from the last run with highest epoch number.

    Args:
        checkpoints_root: Path to the root checkpoints directory.
        max_r: Rotation threshold used in the previous run.
        max_t: Translation threshold used in the previous run.

    Returns:
        Path to the checkpoint file with the highest epoch, or raises FileNotFoundError if none found.
    """
    pattern = f"checkpoint_r{max_r:.2f}_t{max_t:.2f}_e*_*.pth"
    candidates = []
    for path in checkpoints_root.rglob(f"**/models/{pattern}"):
        filename = path.name
        # Extract epoch number: between 'e' and next underscore
        try:
            epoch_str = filename.split("_e")[1].split("_")[0]
            epoch = int(epoch_str)
            candidates.append((epoch, path))
        except (IndexError, ValueError):  # skip malformed names
            continue
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    # Return path with max epoch
    return max(candidates, key=lambda x: x[0])[1]

checkpoints_root = Path("./checkpoints_mixed_scenes/")
print(str(find_best_checkpoint(checkpoints_root, 20.0, 1.5)))
