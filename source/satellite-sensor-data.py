import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load
from datetime import timedelta


delta_t = 30  # Calculate orbit every 30 seconds for one orbit
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
