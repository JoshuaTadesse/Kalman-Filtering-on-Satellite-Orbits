import numpy as np

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