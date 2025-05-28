import numpy as np
from car_dynamics import bicycle_dynamics_discrete
from frenet import get_frenet_coords

class Car:
    def __init__(self, initial_state, v_max, a_max, a_lat_max, max_steering_angle=np.pi/2, dt=0.1, L=2.5):
        self.state = np.array(initial_state, dtype=float)
        self.control_inputs = np.array([0.0, 0.0])  
        self.dt = dt
        self.L = L
        self.history = [self.state.copy()]
        self.v_max = v_max
        self.a_max =  a_max
        self.a_lat_max = a_lat_max
        self.max_steering_angle = max_steering_angle

    def set_control(self, a, delta):
        """Set current control inputs."""
        self.control_inputs = np.array([a, delta])

    def update(self):
        """Advance the state using bicycle dynamics."""
        self.state = bicycle_dynamics_discrete(self.state, self.control_inputs, self.dt, self.L)
        self.history.append(self.state.copy())

    def get_state(self):
        """Return the current state."""
        return self.state
    
    def get_control_inputs(self):
        """Return the current control inputs."""
        return self.control_inputs

    def get_frenet_coords(self, centerline, headings):
        """Return current (s, d) in Frenet frame."""
        x, y = self.state[0], self.state[1]
        return get_frenet_coords(x, y, centerline, headings)
    
    def compute_delta_psi(self, centerline, headings):
        """Compute delta_psi for frenet dynamics"""
        x, y, theta = self.state[0], self.state[1], self.state[2]
        diffs = centerline - np.array([x, y])
        idx = np.argmin(np.sum(diffs**2, axis=1))
        psi_ref = headings[idx]
        delta_psi = theta - psi_ref
        delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi 
        return delta_psi
