import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
import sys

def normalize(v):
    norm = np.linalg.norm(v, axis=0) + 1e-5
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    norm_diff = normalize(np.diff(waypoints, axis=1))
    return np.einsum('ij,ij->j', norm_diff[:, :-1], norm_diff[:, 1:]).sum()

def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    ls_tocenter = np.mean(np.square(waypoints_center - waypoints))
    curv = curvature(waypoints.reshape(2, -1))
    return ls_tocenter - weight_curvature * curv

def waypoint_prediction(roadside1_spline, roadside2_spline, num_waypoints=6, way_type="center"):
    t = np.linspace(0, 1, num_waypoints)
    return (np.array(splev(t, roadside1_spline)) + np.array(splev(t, roadside2_spline))) / 2

def target_speed_prediction(waypoints, num_waypoints_used=5, max_speed=60, exp_constant=4.5, offset_speed=25):
    curv_center = abs(curvature(waypoints[:, :num_waypoints_used]) - num_waypoints_used + 2)
    return (offset_speed + (max_speed - offset_speed) * np.exp(-exp_constant * curv_center)) * 1.25
