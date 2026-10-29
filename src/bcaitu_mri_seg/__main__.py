"""CLI setup."""

import torch
from lightning.pytorch.cli import LightningCLI


def setup() -> None:
    """Perform initialization steps."""
    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")


def main() -> None:
    """Script entry point."""
    setup()
    _cli = LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    main()
