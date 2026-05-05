import io

import numpy as np
import pytest
import torch

from b2_fdm_mppi.core.sequence_fdm_v2 import SequenceFdmMlpV2


def test_model_forward_shapes():
    model = SequenceFdmMlpV2(horizon_steps=10, hidden_dims=[64, 64])
    batch = 4
    state = torch.randn(batch, 6)
    controls = torch.randn(batch, 10, 3)
    terrain = torch.randn(batch, 81)

    states_pred, risk_logits = model(state, controls, terrain)
    assert states_pred.shape == (batch, 10, 6)
    assert risk_logits.shape == (batch, 10)


def test_model_single_sample():
    model = SequenceFdmMlpV2(horizon_steps=5, hidden_dims=[32])
    state = torch.randn(1, 6)
    controls = torch.randn(1, 5, 3)
    terrain = torch.randn(1, 81)

    states_pred, risk_logits = model(state, controls, terrain)
    assert states_pred.shape == (1, 5, 6)
    assert risk_logits.shape == (1, 5)
    assert torch.isfinite(states_pred).all()
    assert torch.isfinite(risk_logits).all()


def test_model_save_load():
    model = SequenceFdmMlpV2(horizon_steps=8, hidden_dims=[64])
    state = {"model_state_dict": model.state_dict(), "horizon_steps": 8, "hidden_dims": [64]}
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)

    model2 = SequenceFdmMlpV2(horizon_steps=loaded["horizon_steps"], hidden_dims=loaded["hidden_dims"])
    model2.load_state_dict(loaded["model_state_dict"])

    x = torch.randn(2, 6)
    u = torch.randn(2, 8, 3)
    g = torch.randn(2, 81)
    s1, r1 = model(x, u, g)
    s2, r2 = model2(x, u, g)
    assert torch.allclose(s1, s2)
    assert torch.allclose(r1, r2)
