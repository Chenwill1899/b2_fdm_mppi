"""High-level learned forward dynamics model (High-Level FDM).

Subpackage layout:

    schema     - constants, dataclasses, frame transforms
    model      - torch.nn.Module
    losses     - loss functions
    trainer    - training / evaluation loops
    rollout    - MPPI integration adapter
    synthetic  - numpy-only synthetic data generator
    dataset    - dataset I/O and builder

Torch-dependent modules (model, losses, trainer, rollout) are imported
lazily so that the numpy-only pieces (schema, synthetic, dataset writer)
can be exercised without torch installed.
"""

from b2_fdm_mppi.high_level_fdm import schema

__all__ = ["schema"]
