import pytest

from b2_fdm_mppi.visualization.utils import animation_axis_limits, map_axis_limits


def test_map_axis_limits_are_fixed_to_forward_twenty_meter_view():
    xlim, ylim = map_axis_limits()

    assert xlim == pytest.approx((0.0, 20.0))
    assert ylim == pytest.approx((-10.0, 10.0))


def test_animation_axis_limits_use_fixed_map_limits():
    xlim, ylim = animation_axis_limits([10.0, 0.0, 0.0])

    assert xlim == pytest.approx((0.0, 20.0))
    assert ylim == pytest.approx((-10.0, 10.0))
