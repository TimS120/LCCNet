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


def run_iteration(script: str, epochs: int, max_t: float, max_r: float, weights: str = None) -> None:
    """Execute one training iteration via Sacred CLI.

    Args:
        script: Path to the training script.
        epochs: Number of epochs to train.
        max_t: Max translation config.
        max_r: Max rotation config.
        weights: Path to weights file or None.

    Raises:
        SystemExit: If the training script returns a non-zero exit code.
    """
    cmd = [sys.executable, script, "with",
           f"epochs={epochs}",
           f"max_t={max_t}",
           f"max_r={max_r}"]
    if weights:
        cmd.append(f"weights=\"{weights}\"")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Training failed on epochs={epochs}, max_t={max_t}, max_r={max_r}")


def main():
    """Main orchestration: checks root folder, then runs sequential iterations."""
    checkpoints_root = Path("./checkpoints/")
    if checkpoints_root.exists():
        sys.exit("Error: './checkpoints/' already exists. Remove or rename it before retrying.")

    # Define iterations: (epochs, max_t, max_r)
    iterations = [
        (120, 1.5, 20.0),
        (50, 1.0, 10.0),
        (50, 0.5, 5.0),
        (50, 0.2, 2.0),
        (50, 0.1, 1.0)
    ]
    script_path = "train_with_sacred.py"
    prev_max_t = None
    prev_max_r = None
    weights = None

    for idx, (epochs, max_t, max_r) in enumerate(iterations):
        if idx == 1:  # We only use the roughest model as pretrained model
            # locate best checkpoint from previous iteration
            weights = str(find_best_checkpoint(checkpoints_root, prev_max_r, prev_max_t))
        run_iteration(script_path, epochs, max_t, max_r, weights)
        if idx == 0:
            prev_max_t, prev_max_r = max_t, max_r


if __name__ == "__main__":
    main()
