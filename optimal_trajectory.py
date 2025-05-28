import numpy as np
from scipy.optimize import minimize

def lap_time_cost_with_smoothness(d, s, a_lat_max=9.0, w_smooth=0.5):
    """Computes the cost that is used in the optimization problem to compute the optimal lateral offsets from the centerline."""
    # Estimation of the lap time
    ds = np.gradient(s)
    d2_ds2 = np.gradient(np.gradient(d, s), s)
    curvature = np.abs(d2_ds2)
    curvature = np.clip(curvature, 1e-4, None)

    v_max = np.sqrt(a_lat_max / curvature)
    segment_times = ds / v_max
    lap_time = np.sum(segment_times)

    # Penalize fast changes in lateral position to have a smooth optimal trajectory
    smoothness = np.sum(np.diff(d, 2)**2)

    total_cost = lap_time + w_smooth * smoothness
    
    return total_cost

def generate_optimal_traj(centerline_frenet, track_width, a_lat_max=9.0):
    """Computes the optimal lateral offsets from the centerline to minimize lap time."""
    N = len(centerline_frenet)
    max_dev = track_width / 2 * 0.95
    np.random.seed(42)
    initial_d = np.random.randn(N)*max_dev
    bounds = [(-max_dev, max_dev)] * N # Ensures that the optimal lateral offsets remain inside of the track

    res = minimize(
        lap_time_cost_with_smoothness,
        initial_d,
        args=(centerline_frenet, a_lat_max),
        bounds=bounds,
        method='L-BFGS-B'
    )

    return res.x