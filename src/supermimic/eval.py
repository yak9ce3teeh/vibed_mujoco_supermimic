"""Deterministic first-frame full-reference evaluation, plus labeled local diagnostics."""

import argparse
import json
from .training import evaluate_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    print(
        json.dumps(evaluate_checkpoint(a.checkpoint, a.output, a.device)["paper_main"]), flush=True
    )


if __name__ == "__main__":
    main()
