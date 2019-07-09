import cv2
import numpy as np

import config


def process_image(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (config.IMG_ROWS, config.IMG_COLS))

    if config.LANE_DETECTION_TYPE == 1:
        return obs

    elif config.LANE_DETECTION_TYPE == 2 or config.LANE_DETECTION_TYPE == 3:
        edges = cv2.Canny(obs, 50, 150)

        rho = 0.8
        theta = np.pi / 180
        threshold = 25
        min_line_len = 5
        max_line_gap = 10

        hough_lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                      maxLineGap=max_line_gap)

        left_lines, right_lines = separate_lines(hough_lines)

        filtered_right, filtered_left = [], []
        if len(left_lines):
            filtered_left = reject_outliers(left_lines, cutoff=(-30.0, -0.1), lane='left')
        if len(right_lines):
            filtered_right = reject_outliers(right_lines, cutoff=(0.1, 30.0), lane='right')

        lines = []
        if len(filtered_left) and len(filtered_right):
            lines = np.expand_dims(np.vstack((np.array(filtered_left), np.array(filtered_right))), axis=0).tolist()
        elif len(filtered_left):
            lines = np.expand_dims(np.expand_dims(np.array(filtered_left), axis=0), axis=0).tolist()
        elif len(filtered_right):
            lines = np.expand_dims(np.expand_dims(np.array(filtered_right), axis=0), axis=0).tolist()

        if config.LANE_DETECTION_TYPE == 2:
            ret_img = np.zeros((80, 80))

            if len(lines):
                try:
                    draw_lines(ret_img, lines, thickness=1)
                except:
                    pass

            return ret_img

        else:
            ret_img = np.zeros((2, 4))
            if len(lines):
                try:
                    ret_img[0] = ((lines[0])[0])[0:4]
                    ret_img[1] = ((lines[0])[1])[0:4]
                except:
                    pass

            return ret_img

    else:
        raise ValueError('Illegal lane_detection type')


def separate_lines(lines):
    right = []
    left = []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            m = slope(x1, y1, x2, y2)
            if m >= 0:
                right.append([x1, y1, x2, y2, m])
            else:
                left.append([x1, y1, x2, y2, m])
    return left, right


def slope(x1, y1, x2, y2):
    try:
        return (y1 - y2) / (x1 - x2)
    except:
        return 0


def reject_outliers(data, cutoff, threshold=0.08, lane='left'):
    data = np.array(data)
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    try:
        if lane == 'left':
            return data[np.argmin(data, axis=0)[-1]]
        elif lane == 'right':
            return data[np.argmax(data, axis=0)[-1]]
    except:
        return []


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2, slope in line:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
