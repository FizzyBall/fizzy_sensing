from fizzy_udp import Fizzy
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import time
import csv
from collections import deque

def extract_euler_from_packet(packet):
    qx, qy, qz, qw = packet[3], packet[4], packet[5], packet[6]
    w, x, y, z = qw, qx, qy, qz

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, pitch, yaw


SIGNALS = {
    "lin_acc": {
        "indices": (7, 8, 9),
        "labels": ("Lin Acc X (g)", "Lin Acc Y (g)", "Lin Acc Z (g)"),
    },
    "raw_acc": {
        "indices": (10, 11, 12),
        "labels": ("Raw Acc X", "Raw Acc Y", "Raw Acc Z"),
    },
    "gyro_raw": {
        "indices": (13, 14, 15),
        "labels": ("Gyro X", "Gyro Y", "Gyro Z"),
    },
}

fizzy = Fizzy()

app = pg.mkQApp("Fizzy View")
win = pg.GraphicsLayoutWidget(show=True, title="Fizzy IMU Data")
win.resize(600, 800)

data_plotting = 100

x_data = [deque(maxlen=data_plotting) for _ in range(4)]
y_data = [deque(maxlen=data_plotting) for _ in range(4)]

x_data_all = []
y_data_all = [[], [], [], []]

MODE = "lin_acc"
indices = SIGNALS[MODE]["indices"]
axis_labels = (*SIGNALS[MODE]["labels"], "Magnitude")

plots = []
curves = []
for i in range(4):
    p = win.addPlot()
    p.setLabel('left', axis_labels[i])
    p.showGrid(x=True, y=True)
    p.enableAutoRange(False)
    p.setYRange(-4, 4)
    c = p.plot(pen='y')
    plots.append(p)
    curves.append(c)
    win.nextRow()

plots[-1].setLabel('bottom', 'Time (s)')

if __name__ == '__main__':
    try:
        while True:
            data = fizzy.get_data()
            if data == -1:
                print("Error: No data received from Fizzy")
                continue

            t = data[0] / 1_000_000
            x, y, z = (data[i] for i in indices)
            mag = np.sqrt(x**2 + y**2 + z**2)
            values = (x, y, z, mag)

            for i in range(4):
                x_data[i].append(t)
                y_data[i].append(values[i])
                curves[i].setData(list(x_data[i]), list(y_data[i]))

            x_data_all.append(t)
            for i, v in enumerate(values):
                y_data_all[i].append(v)

            app.processEvents()

    except KeyboardInterrupt:
        print("Measurement interrupted by user.")

    filename = f"sensor_data_lin_acc_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Time', 'X', 'Y', 'Z', 'Magnitude'])
        for i in range(len(x_data_all)):
            csvwriter.writerow([x_data_all[i], y_data_all[0][i], y_data_all[1][i], y_data_all[2][i], y_data_all[3][i]])
    print(f"Data saved to {filename}")