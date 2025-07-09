import sys
import subprocess
from pathlib import Path


def find_best_checkpoint(checkpoints_root: Path, max_r: float, max_t: float) -> Path:
    """Locate the checkpoint file from the given run with highest epoch number.

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
        raise RuntimeError(
            f"Training failed on epochs={epochs}, max_t={max_t}, max_r={max_r} (return code {result.returncode})"
        )


def main():
    """Main orchestration: checks root folder, then runs sequential iterations."""
    checkpoints_root = Path("./checkpoints/")
    # Define iterations: (epochs, max_t, max_r)
    iterations = [
        (120, 1.5, 20.0),
        (50, 1.0, 10.0),
        (50, 0.5, 5.0),
        (50, 0.2, 2.0),
        (50, 0.1, 1.0)
    ]
    script_path = "train_with_sacred.py"
    failures = []

    # First (roughest) level
    rough_epochs, rough_t, rough_r = iterations[0]
    weights = None
    # Check if roughest-level output already exists
    existing = list(
        checkpoints_root.rglob(
            f"**/models/checkpoint_r{rough_r:.2f}_t{rough_t:.2f}_e*_*.pth"
        )
    )
    if existing:
        print("Iteration 0 (roughest level) already exists, skipping.")
    else:
        try:
            run_iteration(script_path, rough_epochs, rough_t, rough_r, None)
        except Exception as e:
            print(f"Iteration 0 failed: {e}")
            failures.append((0, str(e)))
    # Always load best from roughest level if available
    try:
        best = find_best_checkpoint(checkpoints_root, rough_r, rough_t)
        weights = str(best)
        print(f"Using pretrained weights from roughest level: {weights}")
    except Exception as e:
        print(f"Failed to locate pretrained weights after iteration 0: {e}")
        failures.append(("weights", str(e)))

    # Subsequent finer levels
    for idx, (epochs, max_t, max_r) in enumerate(iterations[1:], start=1):
        matches = list(
            checkpoints_root.rglob(
                f"**/models/checkpoint_r{max_r:.2f}_t{max_t:.2f}_e*_*.pth"
            )
        )
        if matches:
            print(f"Iteration {idx} already exists, skipping.")
            continue
        try:
            run_iteration(script_path, epochs, max_t, max_r, weights)
        except Exception as e:
            print(f"Iteration {idx} failed: {e}")
            failures.append((idx, str(e)))

    # Summary of failures
    if failures:
        print("\nAutomated training completed with FAILURES on following levels:")
        for lvl, msg in failures:
            print(f" - Level {lvl}: {msg}")
    else:
        print("\nAutomated training completed successfully for all levels.")


if __name__ == "__main__":
    main()
