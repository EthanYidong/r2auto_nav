from random import sample
import board
import busio
import adafruit_mlx90640

import numpy as np
import math

# How far left and right to scan and consider. Should be kept the same as the main wallfollow.py script.
THERMAL_ANGLE_BOUNDS = 55.0
# How far up and down to scan and consider. Should be kept the same as the main wallfollow.py script.
THERMAL_H_RANGE = range(8, 16)
# How wide does the detection have to be before the target is considered found? Should be kept the same as the main wallfollow.py script.
SAMPLE_THRESHOLD = 5

# Which percentiles of temperature to report values for
test_percentiles = [50, 60, 70, 80]


# Connect to sensor
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX addr detected on I2C")
        
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
frame = [0] * 768

target = []
no_target = []

# Loop through and scan many different non-target locations at the user's discretion, store the temperature values
while True:
    print("Point at empty space, or target behind a wall, and press enter, or q to continue to targe")
    inp = input()
    if inp == "q":
        break
    mlx.getFrame(frame)
    thermal = np.array(frame).reshape([24,32])
    no_target.append(thermal)


# For each possible percentile, find the "minimum threshold" such that all xth percentile temperatures are under this threshold.
min_threshold = [math.inf for _ in test_percentiles]
for t in no_target:
    for x in range(32):
        temps = []
        for y in THERMAL_H_RANGE:
            temps.append(t[y][x])
        for i, perc in enumerate(test_percentiles):
            min_threshold[i] = min(min_threshold[i], np.percentile(temps, perc))

for i, perc in enumerate(test_percentiles):
    print(f"Minimum threshold for {perc} percentile is {min_threshold[i]}")

# Similarly, loop through and scan many different target locations at the user's discretion, store the temperature values
while True:
    print("Point at target and press enter, or q to end")
    inp = input()
    if inp == "q":
        break
    mlx.getFrame(frame)
    thermal = np.array(frame).reshape([24,32])
    target.append(thermal)

max_threshold = [math.inf for _ in test_percentiles]

# For each possible percentile, find the "maximum threshold" such that using that threshold, the xth percentile heat is at least SAMPLE_THRESHOLD wide.
for t in target:
    sample = [[] for _ in test_percentiles]
    for x in range(32):
        temps = []
        for y in THERMAL_H_RANGE:
            temps.append(t[y][x])
        for i, perc in enumerate(test_percentiles):
            sample[i].append(np.percentile(temps, perc))
    for i, perc in enumerate(test_percentiles):
        sample[i].sort()
        max_threshold[i] = min(max_threshold[i], sample[i][-SAMPLE_THRESHOLD])

for i, perc in enumerate(test_percentiles):
    print(f"Maximum threshold for {perc} percentile is {max_threshold[i]}")
