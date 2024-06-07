import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks
import math

class LaneDetection:
    def __init__(self, cut_size=66, spline_smoothness=10, gradient_threshold=25, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0
        self.distance_threshold = 18

    def cut_gray(self, state_image_full):
        cut_image = state_image_full[:self.cut_size]
        gray_state_image = np.zeros((self.cut_size, 96))
        for i, j in np.ndindex(self.cut_size, 96):
            gray_state_image[i][j] = cut_image[i][j][0] * 0.1140 + cut_image[i][j][1] * 0.5870 + cut_image[i][j][2] * 0.2989
        return gray_state_image[::-1]

    def edge_detection(self, gray_image):
        gradient_sum = np.zeros((self.cut_size, 96))
        gradient_x = np.gradient(gray_image, axis=0)
        gradient_y = np.gradient(gray_image, axis=1)
        #gradient_sum = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_sum = np.abs(gradient_x) + np.abs(gradient_y) #better result
        gradient_sum[gradient_sum < self.gradient_threshold] = 0
        return gradient_sum

    def find_maxima_gradient_rowwise(self, gradient_sum):
        argmaxima = [find_peaks(row, self.distance_maxima_gradient)[0] for row in gradient_sum]
        return argmaxima

    def find_first_lane_point(self, gradient_sum):
        row = 0
        while True:
            argmaxima = find_peaks(gradient_sum[row], distance=3)[0]
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[0,  row]]) if argmaxima[0] < 48 else np.array([[96,  row]])
                return lane_boundary1_startpoint, lane_boundary2_startpoint, True
            if argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                return lane_boundary1_startpoint, lane_boundary2_startpoint, True
            if argmaxima.shape[0] > 2:
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                return lane_boundary1_startpoint, lane_boundary2_startpoint, True
            row += 1
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                return lane_boundary1_startpoint, lane_boundary2_startpoint, False

    def lane_detection(self, state_image_full):
        gray_state = self.cut_gray(state_image_full)
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)
        second_maxima = [[] for _ in range(self.cut_size)]

        if not lane_found:
            return self.lane_boundary1_old, self.lane_boundary2_old

        for i in range(self.cut_size):
            result = [np.linalg.norm(lane_boundary1_points[-1] - np.array([candidates, i])) for candidates in maxima[i]]
            if not result or min(result) >= self.distance_threshold:
                break
            new_point = np.argmin(result)
            lane_boundary1_points = np.vstack([lane_boundary1_points, np.array([maxima[i][new_point], i])])
            second_maxima[i] = np.delete(maxima[i], new_point)
            if not maxima:
                break

        for i in range(self.cut_size):
            result = [np.linalg.norm(lane_boundary2_points[-1] - np.array([candidates, i])) for candidates in second_maxima[i]]
            if not result or min(result) >= self.distance_threshold:
                break
            new_point = np.argmin(result)
            lane_boundary2_points = np.vstack([lane_boundary2_points, np.array([second_maxima[i][new_point], i])])
            if not second_maxima:
                break

        if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
            lane_boundary1,_ = splprep([lane_boundary1_points[1:,0], lane_boundary1_points[1:,1]], s=self.spline_smoothness, k=2)
            lane_boundary2,_ = splprep([lane_boundary2_points[1:,0], lane_boundary2_points[1:,1]], s=self.spline_smoothness, k=2)
        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2
        return lane_boundary1, lane_boundary2

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))
        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        if waypoints:
            plt.scatter(*waypoints, color='white')
        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
