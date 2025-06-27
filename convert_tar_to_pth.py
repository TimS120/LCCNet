#!/usr/bin/env python3
"""
Convert a PyTorch checkpoint saved as a .tar (dict) into a .pth file
containing only the model state_dict.
"""

import argparse
import os

import torch


def convert_checkpoint(input_path: str, output_path: str) -> None:
    """
    Load a checkpoint dict from input_path, extract its 'state_dict',
    and save that dict to output_path.
    """
    checkpoint = torch.load(input_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"'state_dict' not found in checkpoint: {input_path}")
    state_dict = checkpoint["state_dict"]
    torch.save(state_dict, output_path)


def main() -> None:
    """
    Parse command-line arguments and convert one or more checkpoints.
    """
    parser = argparse.ArgumentParser(
        description="Convert .tar checkpoint(s) to raw .pth state_dict file(s)."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to the source .tar checkpoint file(s).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to write the .pth files (default: current directory).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for in_path in args.inputs:
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(args.output_dir, f"{base}.pth")
        convert_checkpoint(in_path, out_path)
        print(f"Converted {in_path} → {out_path}")


if __name__ == "__main__":
    main()
