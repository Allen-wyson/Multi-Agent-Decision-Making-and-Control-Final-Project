import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from racetrack import generate_track
from optimal_trajectory import generate_optimal_traj
from car import Car
from ZS_Controller import zero_sum_best_response_frenet
from frenet import compute_curvature

def resample_centerline(centerline, num_points=500):
    '''Function used to sample additional points along the centerline since trackgen only returns the minimal amount of points necessary to describe the track'''
    ds = np.hypot(np.diff(centerline[:, 0]), np.diff(centerline[:, 1]))
    s = np.insert(np.cumsum(ds), 0, 0.0)
    s_uniform = np.linspace(0, s[-1], num_points)
    interp_x = interp1d(s, centerline[:, 0], kind='linear')
    interp_y = interp1d(s, centerline[:, 1], kind='linear')
    centerline_dense = np.stack([interp_x(s_uniform), interp_y(s_uniform)], axis=1)
    return centerline_dense, s_uniform

# Generate the track
num_points = 1000
Length, Width = 100, 6
track = generate_track(Length, Width)
track.plot()
centerline_raw = np.column_stack((track.xm, track.ym))

# Remove duplicate (x,y) points from the centerline (trackgen can duplicate points sometimes and they can cause numerical errors)
_, unique_indices = np.unique(centerline_raw, axis=0, return_index=True)
centerline_raw = centerline_raw[np.sort(unique_indices)]

# Resample the centerline
centerline, centerline_frenet = resample_centerline(centerline_raw, num_points)

# Compute centerline headings
diffs = np.diff(centerline, axis=0)
headings = np.arctan2(diffs[:, 1], diffs[:, 0])
headings = np.append(headings, headings[-1])

# Initialize cars
g = 9.81
car1 = Car([track.xm[0], track.ym[0]+2, headings[0], 5.0], v_max=65.0, a_max=10.0, a_lat_max=1*g)
car2 = Car([track.xm[0], track.ym[0]-2, headings[0], 5.0], v_max=65.0, a_max=10.0, a_lat_max=2*g)

# Generate the optimal trajectory for each car given the track
d_opt1 = generate_optimal_traj(centerline_frenet, track.width, car1.a_lat_max)
d_opt2 = generate_optimal_traj(centerline_frenet, track.width, car2.a_lat_max)

# Compute the curvature along the centerline (Artificial, only works for the U shaped track)
kappa = np.zeros_like(headings)
straight_angles = [0, np.pi, -np.pi]
tolerance = 1e-2
# Curvature is zero for straight lines (heading is 0 or pi or -pi)
curved_mask = ~np.isclose(headings % (2*np.pi), straight_angles[0], atol=tolerance) & \
              ~np.isclose(headings % (2*np.pi), straight_angles[1], atol=tolerance) & \
              ~np.isclose(headings % (2*np.pi), straight_angles[2], atol=tolerance)

# kappa[curved_mask] = 1.0 / 40.0  # curvature of a semi-circle is simply 1/R

# Curvature interpolation function used in the discrete dynamics
kappa_interp = interp1d(centerline_frenet, kappa, kind='linear', bounds_error=False, fill_value="extrapolate")

# Useful prints for debugging
# print("centerline :", centerline)
# print("headings :", headings)
# print("centerline_frenet :", centerline_frenet)
# print("dopt1 :", d_opt1)
# print("dopt2 :", d_opt2)
# print("curvature:", kappa)

## Simulation
# Simulation parameters init
s_goal = centerline_frenet[-1]
N = 20
dt = 0.05
T = 40

# Cost lists init
J1 = []
J2 = []

# Loop
for t in range(T):
    print(f"Step {t+1}/{T}")
    
    # Get current Frenet coordinates
    s1, _ = car1.get_frenet_coords(centerline, headings)
    s2, _ = car2.get_frenet_coords(centerline, headings)

    # Find the closest index in the centerline
    idx_closest_1 = np.argmin(np.abs(centerline_frenet - s1))
    idx_closest_2 = np.argmin(np.abs(centerline_frenet - s2))

    # Move couple steps forward 
    idx_next_1 = min(idx_closest_1 + int(math.floor(num_points/10)), len(centerline_frenet) - 1)
    idx_next_2 = min(idx_closest_2 + int(math.floor(num_points/10)), len(centerline_frenet) - 1)

    # Next reference point in s
    s_next_1 = centerline_frenet[idx_next_1]
    s_next_2 = centerline_frenet[idx_next_2]

    # Compute centerline curvature at current car position
    kappa_ref_1 = kappa_interp(s1)
    kappa_ref_2 = kappa_interp(s2)
    
    # Compute optimal input for both cars using the itterative best response approach
    u1, u2, cost_J1, cost_J2 = zero_sum_best_response_frenet(
        car1, car2,
        d_opt1, d_opt2, centerline_frenet, kappa_interp, kappa_ref_1, kappa_ref_2,
        centerline, headings, Width, s_goal, s_next_1, s_next_2,
        N=N, dt=dt
    )
    print("u1: ", u1)
    print("u2: ", u2)
    car1.set_control(*u1)
    car2.set_control(*u2)
    car1.update()
    car2.update()
    J1.append(cost_J1)
    J2.append(cost_J2)

# Statistics on cost to assess if saddle point equilibrium is reached or not
J1_array = np.array(J1)
J2_array = np.array(J2)

# Compute absolute difference
diff = np.abs(J1_array - J2_array)

# Compute statistics
mean_diff = np.mean(diff)
median_diff = np.median(diff)
std_diff = np.std(diff)
min_diff = np.min(diff)
max_diff = np.max(diff)

# Print statistics
print("Statistics of |J1 - J2|:")
print(f"Mean   : {mean_diff:.4f}")
print(f"Median : {median_diff:.4f}")
print(f"Std Dev: {std_diff:.4f}")
print(f"Min    : {min_diff:.4f}")
print(f"Max    : {max_diff:.4f}")

# Create box plot
plt.figure(figsize=(6, 5))
plt.boxplot(diff, vert=True, patch_artist=True)
plt.title("Distribution of |J1 - J2|")
plt.ylabel("Absolute Cost Difference")
plt.grid(True)
plt.show()

# Plotting results
h1 = np.array(car1.history)
h2 = np.array(car2.history)

plt.figure(figsize=(10, 8))
plt.plot(track.xm, track.ym, 'k--', label='Centerline')
plt.plot(track.xb1, track.yb1, 'g--', label='Inner Boundary')
plt.plot(track.xb2, track.yb2, 'r--', label='Outer Boundary')
plt.plot(h1[:, 0], h1[:, 1], 'b-', label='Car 1 Trajectory')
plt.plot(h2[:, 0], h2[:, 1], 'orange', label='Car 2 Trajectory')
plt.ylim(-10, 10)
plt.legend()
plt.title("Zero-Sum Racing Simulation")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid(True)
plt.show()