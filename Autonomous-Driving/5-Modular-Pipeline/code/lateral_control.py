import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

EPSILON = 1e-9


class LateralController:
    def __init__(self, gain_constant=5, damping_constant=0.6):
        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0

    def stanley(self, waypoints, speed):
        k = self.gain_constant

        dx = waypoints[0, 1] - waypoints[0, 0]
        dy = waypoints[1, 1] - waypoints[1, 0]
        delta = np.arctan(k * (waypoints[0, 0] - 48) / (speed + EPSILON)) + np.arctan(dx / dy)
        delta_lb, delta_ub = (delta, -delta) if delta < 0 else (-delta, delta)

        damping = (delta - self.previous_steering_angle) * self.damping_constant
        damping = max(min(damping, delta_ub + 0.1 * delta), delta_lb - 0.1 * delta)

        steering_angle = max(min(delta - damping, 0.4), -0.4)

        self.previous_steering_angle = steering_angle

        return steering_angle
