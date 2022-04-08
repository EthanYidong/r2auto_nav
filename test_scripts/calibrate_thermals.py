from random import sample
import board
import busio
import adafruit_mlx90640

import numpy as np
import math

THERMAL_ANGLE_BOUNDS = 55.0
THERMAL_H_RANGE = range(8, 16)
SAMPLE_THRESHOLD = 5

test_percentiles = [50, 60, 70, 80]


i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX addr detected on I2C")
        
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
frame = [0] * 768

target = []
no_target = []

while True:
    print("Point at empty space, or target behind a wall, and press enter, or q to continue to targe")
    inp = input()
    if inp == "q":
        break
    mlx.getFrame(frame)
    thermal = np.array(frame).reshape([24,32])
    no_target.append(thermal)


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

while True:
    print("Point at target and press enter, or q to end")
    inp = input()
    if inp == "q":
        break
    mlx.getFrame(frame)
    thermal = np.array(frame).reshape([24,32])
    target.append(thermal)

max_threshold = [math.inf for _ in test_percentiles]

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
