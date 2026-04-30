import pytest

from b2_fdm_mppi.visualization.utils import animation_axis_limits, map_axis_limits, map_axis_limits_from_config


def test_map_axis_limits_are_fixed_to_forward_twenty_meter_view():
    xlim, ylim = map_axis_limits()

    assert xlim == pytest.approx((0.0, 20.0))
    assert ylim == pytest.approx((-10.0, 10.0))


def test_animation_axis_limits_use_fixed_map_limits():
    xlim, ylim = animation_axis_limits([10.0, 0.0, 0.0])

    assert xlim == pytest.approx((0.0, 20.0))
    assert ylim == pytest.approx((-10.0, 10.0))


def test_map_axis_limits_from_config_uses_map_origin_and_size():
    config = {"simulation": {"map_origin": [5.0, -2.0], "map_size": [100.0, 50.0]}}

    xlim, ylim = map_axis_limits_from_config(config)

    assert xlim == pytest.approx((5.0, 105.0))
    assert ylim == pytest.approx((-2.0, 48.0))
