import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import sys


def normalize(v):
    norm = np.linalg.norm(v, axis=0) + 0.00001
    return v / norm.reshape(1, v.shape[1])


def curvature(waypoints):
    return np.sum(normalize(np.diff(waypoints, axis=1))[:, 1:] * normalize(np.diff(waypoints, axis=1))[:, :-1])


def smoothing_objective(waypoints, waycenter_point, weight_curvature=40):
    ls_tocenter = np.mean((waycenter_point - waypoints) ** 2)
    curv = curvature(waypoints.reshape(2, -1))
    return -1 * weight_curvature * curv + ls_tocenter


def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type="smooth"):
    t = np.linspace(0, 1, num_waypoints)
    points1 = np.array(splev(t, roadside1_spline))
    points2 = np.array(splev(t, roadside2_spline))
    points = np.zeros(shape=(2, num_waypoints))
    if way_type == "center":
        for i, (point1, point2) in enumerate(zip(points1.T, points2.T)):
            points[:, i] = (point1 + point2) / 2
    elif way_type == "smooth":
        for i, (point1, point2) in enumerate(zip(points1.T, points2.T)):
            points[:, i] = (point1 + point2) / 2
        center_point = points.reshape(num_waypoints * 2,)
        points = minimize(smoothing_objective, center_point, args=center_point)["x"]
        points = points.reshape(2, -1)
    return points


def target_speed_prediction(waypoints, num_waypoints_used=5, max_speed=60, min_speed=30, exp_constant=4.5):
    return (max_speed - min_speed) * np.exp(-np.abs(num_waypoints_used - 2 - curvature(waypoints)) * exp_constant) + min_speed
