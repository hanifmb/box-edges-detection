import argparse
import math
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class Line:
    def __init__(self, m, c, inlier_indices=[]):
        self.m = m
        self.c = c
        self.inlier_indices = inlier_indices

    def __str__(self):
        return f"m:{self.m} c:{self.c}"

    def __repr__(self):
        return f"m:{self.m} c:{self.c}"

    def get_line_explicit(self):
        return self.m, self.c

    def get_inlier_count(self):
        return len(self.inlier_indices)


def fit_line(points):
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


def distance_to_line(point, line):
    # Unpack the point coordinates
    x, y = point

    # Calculate the distance from the point to the line
    distance = abs(line.m * x - y + line.c) / math.sqrt(line.m**2 + 1)

    return distance


def sequential_ransac_multi_line_detection(
    data, threshold, min_points, max_iterations, max_lines
):

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
        if best_line.get_inlier_count() <= 5:
            break

        # accumulate the detected line
        best_lines.append(best_line)
        total_inliers_count += best_line.get_inlier_count()

        # remove the inliers
        inlier_indices = []
        # slope, intercept = best_line.get_line_explicit()
        for j, (x, y) in enumerate(remaining_data):
            if distance_to_line((x, y), best_line) < threshold:
                inlier_indices.append(j)

        remaining_data = [
            remaining_data[j]
            for j in range(len(remaining_data))
            if j not in inlier_indices
        ]

        # second stopping condition
        if len(remaining_data) <= 5:
            break

    return best_lines


def ransac_line_detection(data, threshold, min_points, max_iterations):

    best_num_inliers = 0
    for i in range(max_iterations):
        # randomly select a subset of data points
        sample = random.sample(data, min_points)

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
            if distance_to_line((x, y), fitted_line) < threshold:
                inliers.append((x, y))
                curr_num_inliers += 1

        # update the best line model if this model has more inliers
        if curr_num_inliers > best_num_inliers:
            best_line_model = Line(slope, intercept, inliers)
            best_num_inliers = curr_num_inliers

    return best_line_model


def polar_to_cartesian(polar_points):

    cartesian_points = []
    for point in polar_points:
        degrees = point[0]
        radius = point[1]

        radians = math.radians(degrees)
        x_cartesian = radius * math.cos(radians)
        y_cartesian = radius * math.sin(radians)

        cartesian_points.append((x_cartesian, y_cartesian))

    return cartesian_points


def load_points_file(filename):

    points = []
    with open(filename, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, _ = map(float, line.split())
            points.append((x, y))

    return points


def acute_angle_between_lines(m1, m2):
    theta = abs(math.atan(m2) - math.atan(m1))
    theta_degrees = math.degrees(theta)
    return theta_degrees if theta_degrees <= 90 else 180 - theta_degrees


def find_intersection(m1, b1, m2, b2):
    # Calculate the intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return x, y


def distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def find_connected_line_pair(detected_lines):

    # draw circle
    best_intersection = None
    best_inside_circle_cnt = 0
    for i in range(len(detected_lines)):
        for j in range(i + 1, len(detected_lines)):

            line1 = detected_lines[i]
            line2 = detected_lines[j]

            radius = 50
            center = find_intersection(
                line1.m,
                line1.c,
                line2.m,
                line2.c,
            )

            inside_circle_cnt = 0
            # the inlier_idx is not actually an index
            for inlier_idx in line1.inlier_indices:
                if distance(inlier_idx, center) < radius:
                    inside_circle_cnt += 1

            for inlier_idx in line2.inlier_indices:
                if distance(inlier_idx, center) < radius:
                    inside_circle_cnt += 1

            print("inside circle count", inside_circle_cnt)
            print("acute angle", acute_angle_between_lines(line1.m, line2.m))

            if (
                inside_circle_cnt > best_inside_circle_cnt
                and acute_angle_between_lines(line1.m, line2.m) > 85
            ):
                best_intersection = [detected_lines[i], detected_lines[j]]
                best_inside_circle_cnt = inside_circle_cnt

    return best_intersection


def visualize_lines(cartesian_points, best_intersection):

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


def visualize_points_polar(points_polar, lidar, SCAN_RANGE):

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
        "ANGLE_V": [130, 160],
        "ANGLE_H": [170, 210],
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

    # required minimal 2 data points to fit line
    if points_filtered.size < 4:
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
    visualize_lines(cartesian_points, best_line_pair)
    visualize_points_polar(points_filtered, args.lidar, SCAN_RANGE)
    plt.show(block=True)


if __name__ == "__main__":
    main()
