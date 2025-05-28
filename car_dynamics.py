import numpy as np

def bicycle_dynamics_continuous(state, control, L=2.5):
    """Continuous-time bicycle model dynamics."""
    x, y, theta, v = state
    a, delta = control

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v / L * np.tan(delta)
    v_dot = a

    return np.array([x_dot, y_dot, theta_dot, v_dot])


def frenet_bicycle_dynamics(x, u, kappa, L=2.5):
    """Continuous-time bicycle model dynamics in Frenet frame."""
    s, d, v, delta_psi = x
    a, delta = u

    denom = max(1.0 - kappa * d, 1e-5)
    
    s_dot = v * np.cos(delta_psi) / denom
    d_dot = v * np.sin(delta_psi)
    v_dot = a
    delta_psi_dot = v / L * np.tan(delta) - kappa * v * np.cos(delta_psi) / denom

    return np.array([s_dot, d_dot, v_dot, delta_psi_dot])


def rk4_integration(state, control, dt, dynamics):
    """RK4 integration step."""
    k1 = dynamics(state, control)
    k2 = dynamics(state + 0.5 * dt * k1, control)
    k3 = dynamics(state + 0.5 * dt * k2, control)
    k4 = dynamics(state + dt * k3, control)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def bicycle_dynamics_discrete(state, control, dt, L=2.5):
    """Bicycle model dynamics discretized using RK4"""
    x_plus = rk4_integration(state, control, dt, lambda s, u: bicycle_dynamics_continuous(s, u, L))
    return x_plus


def frenet_dynamics_discrete(x, u, kappa_func, dt, L=2.5):
    """Frenet bicycle dynamics discretized using RK4"""
    def dynamics(x_local, u_local):
        s = x_local[0]
        kappa = kappa_func(s)
        return frenet_bicycle_dynamics(x_local, u_local, kappa, L)
    x_plus = rk4_integration(x, u, dt, dynamics)
    return x_plus