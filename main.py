import argparse
import math
import random
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Tuple
from scipy import stats


class Line:
    def __init__(self, m: float, c: float, inlier_points: list = []):
        self.m = m
        self.c = c
        self.inlier_points = inlier_points

    def __str__(self):
        return f"m:{self.m} c:{self.c}"

    def __repr__(self):
        return f"m:{self.m} c:{self.c}"

    def get_line_explicit(self):
        return self.m, self.c

    def get_inlier_count(self):
        return len(self.inlier_points)


def sequential_ransac_multi_line_detection(
    data: npt.NDArray[np.float64],
    threshold: float,
    min_points: int,
    max_iterations: int,
    max_lines: int,
    min_inliers: int = 5,
    min_points_cnt: int = 5,
) -> npt.NDArray[Line]:

    best_lines = []
    remaining_data = data

    total_inliers_count = 0
    for i in range(max_lines):

        best_line = ransac_line_detection(
            data=remaining_data,
            threshold=threshold,
            min_points=min_points,
            max_iterations=max_iterations,
        )

        # first stopping condition
        if best_line.get_inlier_count() <= min_inliers:
            break

        # fit the new line
        X = np.array(best_line.inlier_points)[:, 0]
        Y = np.array(best_line.inlier_points)[:, 1]
        slope, intercept, _, _, _ = stats.linregress(X, Y)

        best_line_fit = Line(slope, intercept)
        best_line_fit.inlier_points = best_line.inlier_points

        # accumulate the fitted line
        best_lines.append(best_line_fit)
        total_inliers_count += best_line.get_inlier_count()

        # remove the inliers
        inlier_points = []
        for j, (x, y) in enumerate(remaining_data):
            if calc_dist_to_line((x, y), best_line) < threshold:
                inlier_points.append(j)

        remaining_data = np.delete(remaining_data, inlier_points, axis=0)

        # second stopping condition
        if len(remaining_data) <= min_points_cnt:
            break

    return np.array(best_lines)


def ransac_line_detection(
    data: npt.NDArray[np.float64],
    threshold: float,
    min_points: int,
    max_iterations: int,
) -> Line:

    best_num_inliers = 0
    for i in range(max_iterations):
        # randomly select a subset of data points
        sample = data[np.random.choice(data.shape[0], 2, replace=False), :]

        # fit a line to the subset of data points
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        if (
            x2 - x1 == 0
        ):  # if the two sampled points have the same x-coordinate, skip this iteration
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        fitted_line = Line(slope, intercept)

        # count the number of inliers (data points that are within the threshold distance of the line)
        curr_num_inliers = 0
        inliers = []
        for idx, (x, y) in enumerate(data):
            if calc_dist_to_line((x, y), fitted_line) < threshold:
                inliers.append((x, y))
                curr_num_inliers += 1

        # update the best line model if this model has more inliers
        if curr_num_inliers > best_num_inliers:
            best_line_model = Line(slope, intercept, inliers)
            best_num_inliers = curr_num_inliers

    return best_line_model


def fit_line(points: list):
    n = len(points)

    # Calculate the mean of x and y
    x_mean = sum([point[0] for point in points]) / n
    y_mean = sum([point[1] for point in points]) / n

    # Calculate the slope and y-intercept of the linelower
    numerator = sum(
        [(points[i][0] - x_mean) * (points[i][1] - y_mean) for i in range(n)]
    )
    denominator = sum([(points[i][0] - x_mean) ** 2 for i in range(n)])
    m = numerator / denominator
    c = y_mean - m * x_mean

    # Return the slope and y-intercept
    return m, c


def calc_dist_to_line(point: Tuple[float, float], line: Line):
    # Unpack the point coordinates
    x, y = point

    # Calculate the distance from the point to the line
    distance = abs(line.m * x - y + line.c) / math.sqrt(line.m**2 + 1)

    return distance


def calc_dist_to_point(point1: Tuple[float, float], point2: Tuple[float, float]):

    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def polar_to_cartesian(
    polar_points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:

    cartesian_points = []
    for point in polar_points:
        degrees = point[0]
        radius = point[1]

        radians = math.radians(degrees)
        x_cartesian = radius * math.cos(radians)
        y_cartesian = radius * math.sin(radians)

        cartesian_points.append((x_cartesian, y_cartesian))

    return np.array(cartesian_points)


def load_points_file(filename):

    points = []
    with open(filename, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, _ = map(float, line.split())
            points.append((x, y))

    return points


def calc_acute_angle(line1: Line, line2: Line):

    theta = abs(math.atan(line2.m) - math.atan(line1.m))
    theta_degrees = math.degrees(theta)
    return theta_degrees if theta_degrees <= 90 else 180 - theta_degrees


def find_intersection(line1: Line, line2: Line):
    # Calculate the intersection point
    x = (line2.c - line1.c) / (line1.m - line2.m)
    y = line1.m * x + line1.c

    return x, y


def find_connected_line_pair(detected_lines: npt.NDArray[Line]) -> Tuple[Line, Line]:

    # draw circle
    best_inside_circle_cnt = 0
    for i in range(len(detected_lines)):
        for j in range(i + 1, len(detected_lines)):

            line1 = detected_lines[i]
            line2 = detected_lines[j]

            radius = 50
            center = find_intersection(line1, line2)

            inside_circle_cnt = 0

            for inlier_point in line1.inlier_points:
                if calc_dist_to_point(inlier_point, center) < radius:
                    inside_circle_cnt += 1

            for inlier_point in line2.inlier_points:
                if calc_dist_to_point(inlier_point, center) < radius:
                    inside_circle_cnt += 1

            if (
                inside_circle_cnt > best_inside_circle_cnt
                and calc_acute_angle(line1, line2) > 85
            ):
                best_intersection = (detected_lines[i], detected_lines[j])
                best_inside_circle_cnt = inside_circle_cnt

    return best_intersection


def visualize_lines(
    cartesian_points: npt.NDArray[np.float64], best_intersection: Tuple[Line, ...]
):

    x_coords = [p[0] for p in cartesian_points]
    y_coords = [p[1] for p in cartesian_points]

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0)

    ax = fig.add_subplot()

    ax.scatter(x_coords, y_coords)
    ax.set_xlim(-2000, 1000)
    ax.set_ylim(-2000, 1000)

    # draw the lines
    for line in best_intersection:
        x = np.array([-5000, 5000])
        y = line.m * x + line.c
        ax.plot(x, y)


def visualize_points_polar(
    points_polar: npt.NDArray[np.float64], lidar: str, SCAN_RANGE: dict
):

    thetas = [p[0] for p in points_polar]
    rhos = [p[1] for p in points_polar]

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(wspace=0)
    ax = fig.add_subplot(polar=True)

    if lidar == "horizontal":
        ax.set_xlim(
            np.radians(SCAN_RANGE["ANGLE_H"][0]), np.radians(SCAN_RANGE["ANGLE_H"][1])
        )
    elif lidar == "vertical":
        ax.set_xlim(
            np.radians(SCAN_RANGE["ANGLE_V"][0]), np.radians(SCAN_RANGE["ANGLE_V"][1])
        )

    ax.set_ylim(0, SCAN_RANGE["DIST_H"])
    ax.scatter(np.radians(thetas), rhos)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="1h.txt")
    parser.add_argument("--lidar", default="horizontal")
    parser.add_argument("--threshold", default=5, type=float)
    parser.add_argument("--iter", default=1000, type=int)
    args = parser.parse_args()

    SCAN_RANGE = {
        "ANGLE_V": (130, 160),
        "ANGLE_H": (170, 210),
        "DIST_V": 2000,
        "DIST_H": 2000,
    }

    points = np.array(load_points_file(args.file))

    if args.lidar == "horizontal":
        mask = (
            (points[:, 0] > SCAN_RANGE["ANGLE_H"][0])
            & (points[:, 0] < SCAN_RANGE["ANGLE_H"][1])
            & (np.absolute(points[:, 1]) < SCAN_RANGE["DIST_H"])
        )
    elif args.lidar == "vertical":
        mask = (
            (points[:, 0] > SCAN_RANGE["ANGLE_V"][0])
            & (points[:, 0] < SCAN_RANGE["ANGLE_V"][1])
            & (np.absolute(points[:, 1]) < SCAN_RANGE["DIST_V"])
        )
    else:
        print("lidar type is not recognized")
        return

    points_filtered = points[mask]

    # four data points are required to fit two lines
    if points_filtered.size < 8:
        print(f"The number of input points are too small: {points_filtered.size}")
        return

    cartesian_points = polar_to_cartesian(points_filtered)

    detected_lines = sequential_ransac_multi_line_detection(
        cartesian_points,
        threshold=args.threshold,
        min_points=2,
        max_iterations=args.iter,
        max_lines=3,
    )

    # find the line pair denoting the two edges of the box
    best_line_pair = find_connected_line_pair(detected_lines)

    # visualization
    # visualize_lines(cartesian_points, detected_lines)
    visualize_lines(cartesian_points, best_line_pair)
    visualize_points_polar(points_filtered, args.lidar, SCAN_RANGE)
    plt.show(block=True)


if __name__ == "__main__":
    main()
