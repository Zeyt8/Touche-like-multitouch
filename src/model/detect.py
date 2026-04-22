import serial
import time
import re
import os
from collections import deque
import matplotlib.pyplot as plt

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
WINDOW_SIZE = 10
N_POINTS = 5
POINT_DURATION_SECONDS = 5
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'data.txt')
PLOT_FILE = os.path.join(os.path.dirname(__file__), 'measurements_plot.png')

def acquire_point(ser):
    """Acquire one point for POINT_DURATION_SECONDS and return final moving average per frequency."""
    moving_averages = {}
    start_time = time.time()

    while time.time() - start_time < POINT_DURATION_SECONDS:
        if ser.in_waiting <= 0:
            continue

        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        match = re.match(r'\((\d+),(\d+)\)', line)
        if not match:
            continue

        frequency = int(match.group(1))
        analog_value = int(match.group(2))

        if frequency not in moving_averages:
            moving_averages[frequency] = deque(maxlen=WINDOW_SIZE)

        moving_averages[frequency].append(analog_value)

    point_values = {}
    for frequency, values in moving_averages.items():
        point_values[frequency] = sum(values) / len(values)

    return point_values


def save_measurements(measurements):
    with open(OUTPUT_FILE, 'w') as f:
        for point_values in measurements:
            row = []
            for _, value in point_values.items():
                row.append(f"{value:.0f}")
            f.write(' '.join(row) + '\n')


def plot_measurements(measurements, output_file=PLOT_FILE):
    if not measurements:
        return

    frequencies = sorted({freq for point_values in measurements for freq in point_values.keys()})

    plt.figure(figsize=(10, 6))
    for idx, point_values in enumerate(measurements, start=0):
        y_values = [point_values.get(freq) for freq in frequencies]
        plt.plot(frequencies, y_values, marker='o', linewidth=1.5, label=f"Point {idx}")

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Value')
    plt.title('Measurements per Point')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


if __name__ == "__main__":
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")

    measurements = []

    for point_idx in range(N_POINTS):
        print(f"\nAcquiring point {point_idx}/{N_POINTS} for {POINT_DURATION_SECONDS} seconds...")
        point_values = acquire_point(ser)
        measurements.append(point_values)

    save_measurements(measurements)
    plot_measurements(measurements)

    ser.close()