import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, pi
from skyfield.api import load
from datetime import timedelta

# Constants
mu = 3.986e14  # Gravitational parameter (m^3/s^2)
Re = 6378100  # Earth radius in m
J2 = 1.08263e-3  # J2 perturbation constant
Cd = 2.6  # Drag coefficient
A = 1512.00  # Drag cross-sectional area in m^2 
mass = 460460  # Mass of the ISS in kg
earth_rotation_rate = 7.2921159e-5  # rad/s 

# Atmospheric density model
def atmospheric_density(altitude):
    # Approximate atmospheric density model (kg/m^3)
    altitude = altitude / 1000
    return 1.02e07 * altitude**-7.172

# Calculate J2 perturbation effect
def j2_perturbation(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    factor = -1.5 * J2 * mu * (Re**2) / (r**5)
    z2 = z**2 / r**2
    pert_x = factor * x * (5 * z2 - 1)
    pert_y = factor * y * (5 * z2 - 1)
    pert_z = factor * z * (5 * z2 - 3)
    return np.array([pert_x, pert_y, pert_z])

# Atmospheric drag acceleration
def calculate_drag(x, y, z, v_x, v_y, v_z):
    altitude = np.sqrt(x**2 + y**2 + z**2) - Re
    rho = atmospheric_density(altitude)
    drag_acc_x = -0.5 * Cd * (A / mass) * rho * v_x * np.sqrt(v_x**2 + v_y**2 + v_z**2)
    drag_acc_y = -0.5 * Cd * (A / mass) * rho * v_y * np.sqrt(v_x**2 + v_y**2 + v_z**2)
    drag_acc_z = -0.5 * Cd * (A / mass) * rho * v_z * np.sqrt(v_x**2 + v_y**2 + v_z**2)
    
    return np.array([drag_acc_x, drag_acc_y, drag_acc_z])

# Gravitational, drag, and J2 perturbation acceleration
def acceleration_calculation(x, y, z, v_x, v_y, v_z):
    r = np.sqrt(x**2 + y**2 + z**2)
    g_x = -mu * x / r**3
    g_y = -mu * y / r**3
    g_z = -mu * z / r**3
    j2_effect = j2_perturbation(x, y, z)
    drag_accel = calculate_drag(x, y, z, v_x, v_y, v_z)
    return np.array([g_x + j2_effect[0] + drag_accel[0], g_y + j2_effect[1] + drag_accel[1], g_z + j2_effect[2] + drag_accel[2]])

# RK4 step with refined gravitational model
def rk4_step(state, delta_t):
    def derivatives(s):
        x, y, z, v_x, v_y, v_z = s
        acceleration = acceleration_calculation(x, y, z, v_x, v_y, v_z)
        return np.array([
            v_x, v_y, v_z,
            acceleration[0], acceleration[1], acceleration[2]
        ])

    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * delta_t * k1)
    k3 = derivatives(state + 0.5 * delta_t * k2)
    k4 = derivatives(state + delta_t * k3)

    new_state = state + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state

delta_t = 30  # Calculate orbit every 30 seconds for one orbit
time_steps = int((94 * 60) / delta_t)
initial_state = np.array([
    3956.523018829280 * 1000,   # Initial x (m)
    -5379.232414228390 * 1000,  # Initial y (m)
    -1280.075176778060 * 1000,  # Initial z (m)
    4.56157890161865 * 1000,    # Initial x-velocity (m/s)
    1.97789913886447 * 1000,    # Initial y-velocity (m/s)
    5.82140814804799 * 1000,    # Initial z-velocity (m/s)
])

# Simulate orbit
current_state = initial_state
positions_calculated = np.zeros((time_steps, 3))
positions_calculated[0] = np.array([initial_state[0], initial_state[1], initial_state[2]])
for i in range(1, time_steps):
    current_state = rk4_step(current_state, delta_t)
    positions_calculated[i] = current_state[0:3]

steps = int(5640 / delta_t)

# Load TLE data for the ISS
stations_url = 'https://celestrak.com/NORAD/elements/stations.txt'
satellites = load.tle_file(stations_url)
iss = {sat.name: sat for sat in satellites}['ISS (ZARYA)']

# Generate the measured data using Skyfield
ts = load.timescale()
t = ts.utc(2024, 11, 9, 12, 52, 0)
positions_measured = []

for i in range(steps):
    t2 = t + timedelta(seconds=i * delta_t)
    geocentric = iss.at(t2)
    subpoint = geocentric.subpoint()
    pos = geocentric.position.km * 1000
    positions_measured.append(pos)
positions_measured = np.array(positions_measured)
noise_std = 120000  # Noise level (standard deviation)

# Add random noise to the measurements
noisy_positions_measured_x = positions_measured[:, 0] + np.random.normal(20000, noise_std, len(positions_measured[:, 0]))
noisy_positions_measured_y = positions_measured[:, 1] + np.random.normal(20000, noise_std, len(positions_measured[:, 1]))
noisy_positions_measured_z = positions_measured[:, 2] + np.random.normal(20000, noise_std, len(positions_measured[:, 2]))

# Add some bias
noisy_positions_measured_x1 = noisy_positions_measured_x[:90] + 70000
noisy_positions_measured_x2 = noisy_positions_measured_x[90:] - 120000
noisy_positions_measured_y1 = noisy_positions_measured_y[:90] + 100000
noisy_positions_measured_y2 = noisy_positions_measured_y[90:] - 80000
noisy_positions_measured_z1 = noisy_positions_measured_z[:90] + 50000
noisy_positions_measured_z2 = noisy_positions_measured_z[90:] - 100000

noisy_positions_measured_xf = np.concatenate((noisy_positions_measured_x1, noisy_positions_measured_x2))
noisy_positions_measured_yf = np.concatenate((noisy_positions_measured_y1, noisy_positions_measured_y2))
noisy_positions_measured_zf = np.concatenate((noisy_positions_measured_z1, noisy_positions_measured_z2))

# Constants for EKF
num_steps = positions_calculated.shape[0]
state_dim = 6  # [x, y, z, vx, vy, vz]

# Initialize state covariance matrices, measurement and process noise
state_estimate = initial_state
P = np.eye(state_dim) * 1e6 # Initial large uncertainty in estimate
process_noise_cov = np.diag([5e0, 5e0, 5e0, 1e-3, 1e-3, 1e-3])
measurement_noise_cov = np.diag([100*noise_std**2, 100*noise_std**2, 100*noise_std**2])

# State transition function using RK4 integration step
def transition_function(state):
    return rk4_step(state, delta_t)

# Jacobian of the state transition function (linearized approximation of transition function)
def transition_jacobian(state):
    epsilon = 1e-6  # Small perturbation for numerical Jacobian
    F = np.eye(state_dim)

    for i in range(state_dim):
        perturbed_state = state.copy()
        perturbed_state[i] += epsilon
        F[:, i] = (transition_function(perturbed_state) - transition_function(state)) / epsilon
    return F

# Measurement function (observing position)
def measurement_function(state):
    return state[:3]

# Jacobian of the measurement function
def measurement_jacobian(state):
    H = np.zeros((3, state_dim))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    return H

# EKF Prediction and Update steps
state_estimates = np.zeros((num_steps, state_dim))
state_estimates[0] = state_estimate

for i in range(1, num_steps):
    # Prediction step
    F = transition_jacobian(state_estimate)
    state_predict = transition_function(state_estimate)
    P_predict = F @ P @ F.T + process_noise_cov

    # Measurement update step
    H = measurement_jacobian(state_predict)
    z = np.array([noisy_positions_measured_xf[i], noisy_positions_measured_yf[i], noisy_positions_measured_zf[i]])  # Noisy measurements
    y = z - measurement_function(state_predict)  # Residual
    S = H @ P_predict @ H.T + measurement_noise_cov  # Residual covariance
    K = P_predict @ H.T @ np.linalg.inv(S)  # Kalman gain
    state_estimate = state_predict + K @ y  # Updated state estimate
    P = (np.eye(state_dim) - K @ H) @ P_predict  # Updated covariance

    # Store estimated state
    state_estimates[i] = state_estimate

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the orbits and estimated trajectory
ax.plot(positions_calculated[:, 0], positions_calculated[:, 1], positions_calculated[:, 2], label='ISS Orbit Calculated', color='blue')
ax.plot(noisy_positions_measured_xf, noisy_positions_measured_yf, noisy_positions_measured_zf, label='ISS Orbit Measured', color='lime')
ax.plot(state_estimates[:, 0], state_estimates[:, 1], state_estimates[:, 2], label='EKF Estimated Orbit', color='red')
ax.scatter(0, 0, 0, color="brown", label="Earth Center")
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.title('Effect of Kalman Filtering on Smoothing Out Noisy Trajectory Data of the ISS')
plt.show()
