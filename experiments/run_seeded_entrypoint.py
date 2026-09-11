"""Run a Python entry point after explicitly seeding process-level RNGs."""

from __future__ import annotations

import argparse
import random
import runpy
import sys
from pathlib import Path


def seed_process(seed: int) -> tuple[float, float]:
    """Seed Python and NumPy and return a deterministic audit witness."""

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    return random.random(), float(np.random.random())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--process-seed", type=int, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    seed_process(args.process_seed)
    sys.argv = [str(args.entrypoint), *remainder]
    runpy.run_path(str(args.entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
