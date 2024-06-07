import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

class LongitudinalController:
    def __init__(self, KP=0.01, KI=0.0, KD=0.05):
        self.last_error = 0
        self.sum_error = 0
        self.last_control = 0
        self.speed_history = []
        self.target_speed_history = []
        self.step_history = []

        self.KP = KP
        self.KI = KI
        self.KD = KD

    def PID_step(self, speed, target_speed):
        err = target_speed - speed
        self.sum_error = max(min(self.sum_error + err, 1), -1)

        PID = self.KP * err + self.KI * self.sum_error + self.KD * (err - self.last_error)

        self.last_error = err
        return PID

    def control(self, speed, target_speed):
        control_signal = self.PID_step(speed, target_speed)
        brake, gas = 0, 0

        if control_signal >= 0:
            gas = min(control_signal, 0.8)
        else:
            brake = min(-control_signal, 0.8)

        return gas, brake

    def plot_speed(self, speed, target_speed, step, fig):
        self.speed_history.append(speed)
        self.target_speed_history.append(target_speed)
        self.step_history.append(step)

        plt.gcf().clear()
        plt.plot(self.step_history, self.speed_history, color='green')
        plt.plot(self.step_history, self.target_speed_history, linestyle='--')
        fig.canvas.flush_events()
