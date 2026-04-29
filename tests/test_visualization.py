import pytest

from b2_fdm_mppi.visualization.utils import animation_axis_limits


def test_animation_axis_limits_keep_wide_y_view_for_goal_on_x_axis():
    xlim, ylim = animation_axis_limits([10.0, 0.0, 0.0])

    assert xlim == pytest.approx((-1.0, 11.0))
    assert ylim == pytest.approx((-10.0, 10.0))
