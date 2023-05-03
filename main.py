import argparse
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

x_coords = None
y_coords = None
x_coords_sub = None
y_coords_sub = None
xy_coords_sub = None

def fit_line(points):
    n = len(points)

    # Calculate the mean of x and y
    x_mean = sum([point[0] for point in points]) / n
    y_mean = sum([point[1] for point in points]) / n

    # Calculate the slope and y-intercept of the line
    numerator = sum([(points[i][0] - x_mean) * (points[i][1] - y_mean) for i in range(n)])
    denominator = sum([(points[i][0] - x_mean) ** 2 for i in range(n)])
    m = numerator / denominator
    c = y_mean - m * x_mean

    # Return the slope and y-intercept
    return m, c

def distance_to_line(point, slope, intercept):
    # Unpack the point coordinates
    x, y = point
    
    # Calculate the distance from the point to the line
    distance = abs(slope * x - y + intercept) / math.sqrt(slope ** 2 + 1)
    
    return distance

def sequential_ransac_multi_line_detection(data, threshold, min_points, max_iterations, num_of_lines_to_detect):

    best_lines = []
    color_choice = [(1, 0, 0, 0.7), (0, 1, 0, 0.7), (1, 0, 1, 0.7), (1, 0, 0, 0.7), (0, 1, 0, 0.7), (1, 0, 1, 0.7)]
    color = [(0, 0, 1, 0.7)] * len(data)
    remaining_data = data

    for i in range(num_of_lines_to_detect):
        best_line = ransac_line_detection(data=remaining_data, threshold=threshold, min_points=min_points, max_iterations=max_iterations)

        slope, intercept = best_line
        inlier_indices = []
        for j, (x, y) in enumerate(remaining_data):
            if distance_to_line((x, y), slope, intercept) < threshold:
                inlier_indices.append(j)

        remaining_data = [remaining_data[j] for j in range(len(remaining_data)) if j not in inlier_indices]

        best_lines.append(best_line)
    
    # print("best_lines: ", best_lines)
    # iterate through all the best lines: assign colors and estimate new line on the inliers
    color_code = 0
    new_lines = []

    print("slope(m) intersection(c) color(r, g, b, a)")
    for best_line in best_lines:
        slope, intercept = best_line
        inlier_points = []

        for idx, (x, y) in enumerate(data):
            if distance_to_line((x, y), slope, intercept) < threshold:
                color[idx] = color_choice[color_code]
                inlier_points.append((x, y))

        m, c = fit_line(inlier_points)
        new_lines.append((m, c))

        print(m, c, color_choice[color_code])
        color_code += 1

    # print("best_lines:", new_lines)

    return color, new_lines
    
def ransac_line_detection(data, threshold, min_points, max_iterations):

    best_line_model = None
    best_num_inliers = 0
    for i in range(max_iterations):
        # randomly select a subset of data points
        sample = random.sample(data, min_points)
        
        # fit a line to the subset of data points
        x1, y1 = sample[0]
        x2, y2 = sample[1]
        if x2 - x1 == 0: # if the two sampled points have the same x-coordinate, skip this iteration
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # count the number of inliers (data points that are within the threshold distance of the line)
        num_inliers = 0
        for x, y in data:
            if distance_to_line((x, y), slope, intercept) < threshold:
                num_inliers += 1
        
        # update the best line model if this model has more inliers
        if num_inliers > best_num_inliers:
            best_line_model = (slope, intercept)
            best_num_inliers = num_inliers
    
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
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            x, y, _ = map(float, line.split())
            points.append((x, y))

    return points


def onselect(eclick, erelease):
    # Get the coordinates of the selected rectangle

    global x_coords_sub
    global y_coords_sub
    global xy_coords_sub

    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    # Get the indices of the points within the selected rectangle
    ind = np.where((x_coords >= min(x1,x2)) & (x_coords <= max(x1,x2)) & (y_coords >= min(y1,y2)) & (y_coords <= max(y1,y2)))[0]

    x_coords_sub = [x_coords[i] for i in ind]
    y_coords_sub = [y_coords[i] for i in ind]
    xy_coords_sub = [(x_coords_sub[i], y_coords_sub[i]) for i in range(len(x_coords_sub))]

    # Print the indices of the selected points
    # print('Selected points:', xy_coords_sub)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default="1v.txt")
    parser.add_argument('--threshold', default=2, type=float)
    parser.add_argument('--iter', default=1000, type=int)
    args = parser.parse_args()

    points = load_points_file(args.file)
    cartesian_points = polar_to_cartesian(points)

    global x_coords 
    global y_coords 

    x_coords = [p[0] for p in cartesian_points]
    y_coords = [p[1] for p in cartesian_points]
  
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter(x_coords, y_coords, c=[[0, 0, 1, 0.7]])
    # plt.scatter(x_coords, y_coords, c = color)

    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], 
                                       minspanx=5, minspany=5, spancoords='data')

    plt.xlim([-1500, 500])
    plt.ylim([-2000, 1500])
    plt.draw()

    executed = False
    while plt.fignum_exists(fig.number):
        if xy_coords_sub is not None and not executed:
            color, detected_lines = sequential_ransac_multi_line_detection(xy_coords_sub, threshold=args.threshold, min_points=2, max_iterations=args.iter, num_of_lines_to_detect=2)
            sc.set_offsets(np.c_[x_coords_sub,y_coords_sub])
            sc.set_facecolor(color)
            executed = True

        fig.canvas.draw_idle()
        plt.pause(0.1)

    # plt.waitforbuttonpress()

if __name__ == "__main__":
    main()
