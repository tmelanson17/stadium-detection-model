import cv2 
import numpy as np

import os

MAX_ANGLE_FROM_VERTICAL = 10
MAX_ANGLE_FROM_HORIZONTAL = 5
T1 = 200
T2 = 700
PIXEL_RES = 1
ANGLE_RES = np.pi / 180
THRESHOLD = 10
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 5

def get_line_img(img, lines_fine):
    line_img = np.zeros_like(img)
    if lines_fine is None:
        return line_img
    for line in lines_fine:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 != 0:
                angle = np.arctan((y2 - y1) / (x2 - x1))
                angle_degrees = np.degrees(angle)
            else:
                angle_degrees = 90  # Vertical line

            if (abs(angle_degrees) > MAX_ANGLE_FROM_HORIZONTAL and abs(angle_degrees) < 180 - MAX_ANGLE_FROM_HORIZONTAL) and \
                (abs(angle_degrees - 90) > MAX_ANGLE_FROM_VERTICAL and abs(angle_degrees + 90) > MAX_ANGLE_FROM_VERTICAL):
                continue # Not using diagonal lines
        
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White for all lines
    return line_img

def predict(input_img):
    # Do histogram EQ, find edges
    equalized_img = cv2.equalizeHist(input_img)
    edges = cv2.Canny(equalized_img, threshold1=T1, threshold2=T2)
    # Find lines
    lines = cv2.HoughLinesP(edges, PIXEL_RES, ANGLE_RES, THRESHOLD, minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    # Get line image
    line_img = get_line_img(input_img, lines)
    # Find contours
    contours, hierarchy = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Copy image, draw on it
    original_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2)  # Draw contours in green
    return original_img

if __name__ == "__main__":
    IMG_DIR= '/home/tmelanson/battle4'  # Adjust this path as needed
    img_keyword = 'frame' # Adjust this keyword based on your image naming convention

    # Example usage
    for i in range(100, 2000):
        img_file = os.path.join(IMG_DIR, f'{img_keyword}_{i:06d}.png')
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # Replace with your image path
        if img is None:
            print("Error: Image not found.")
            result = np.zeros_like(img)
        else:
            result = predict(img)