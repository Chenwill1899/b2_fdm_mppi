| mode | controller | params | main_gain | tradeoff | paper_use |
| --- | --- | --- | --- | --- | --- |
| Conservative / smooth | Default learned | residual_gain=1.0 | lower terrain risk, smoother control, lower jerk | worse final distance and more steps than nominal | risk/smoothness operating point |
| Efficiency | Efficiency | residual_gain=0.5, goal_xy_weight=3.5, smooth_weight=1.0 | best official final-distance and steps gains | small terrain-risk, smoothness, and jerk penalties | navigation efficiency operating point |
| Balanced | Balanced | residual_gain=0.5, goal_xy_weight=3.0, smooth_weight=0.75 | steps improve while final distance and risk stay near nominal | small smoothness and jerk penalties | balanced operating-point candidate |
