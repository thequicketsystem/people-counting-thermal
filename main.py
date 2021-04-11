import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2
from math import sqrt

# Sources:
# https://answers.opencv.org/question/210645/detection-of-people-from-above-with-thermal-camera/
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/
# https://github.com/thequicketsystem/people-counting-visual/
# https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

IMG_WIDTH, IMG_HEIGHT = 32, 24
TEMP_MIN, TEMP_MAX = 6, 20

SCALE_FACTOR = 10

# yeah i know they aren't quadrants if there's only two but we'll get to that
QUAD_SEP = (IMG_WIDTH * SCALE_FACTOR) // 2

# no magic numbers
RESULT_COUNT_INDEX = 0
LEFT_QUAD_INDEX = 1
RIGHT_QUAD_INDEX = 2

i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

f = [0] * (IMG_WIDTH * IMG_HEIGHT)

## Blob detection parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 600
params.maxArea = 7000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

def get_best_of_x(x: int) -> int:
    # we don't really need the count from get_frame_data() but we'll keep it for
    # now.
    result = [False, False]

    for i in range(x):
        data = get_frame_data()
        if data[LEFT_QUAD_INDEX]:
            result[LEFT_QUAD_INDEX - 1] = True
        
        if data[RIGHT_QUAD_INDEX]:
            result[RIGHT_QUAD_INDEX - 1] = True

        # the result isn't going to change again if both are true, so break
        # if/when that occurs
        if all(result):
            break

    return result.count(True)

# TODO: Look up the type hinting for this
# Should return a list like this: [count, left_quad, right_quad]
def get_frame_data():

    result = [0, False, False]

    try:
        mlx.getFrame(f)
    except ValueError:
        pass

    temp_data = np.array(f).reshape((IMG_HEIGHT, IMG_WIDTH))

    temp_data = cv2.resize(temp_data, dsize=(IMG_WIDTH * SCALE_FACTOR, IMG_HEIGHT * SCALE_FACTOR))
    temp_data = cv2.normalize(temp_data, temp_data, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # drop colder temp data
    temp_data[temp_data < 80] = 0

    # smoothes image and reduces noise while preserving edges
    temp_data = cv2.bilateralFilter(temp_data, 9, 150, 150)

    kernel = np.ones((5,5), np.uint8)

    temp_data = cv2.erode(temp_data, kernel, iterations = 1)
    temp_data = cv2.dilate(temp_data, kernel, iterations = 1)

    temp_data = cv2.morphologyEx(temp_data, cv2.MORPH_CLOSE, kernel)

    temp_data = cv2.bitwise_not(temp_data)

    keypoints = detector.detect(temp_data)

    result[RESULT_COUNT_INDEX] = len(keypoints)

    # Determine "quadrants" (only two quads for now) of keypoints
    pts = cv2.KeyPoint_convert(keypoints)
    for point in pts:
        x, _ = point.pt
        if x < QUAD_SEP:
            result[LEFT_QUAD_INDEX] = True
        else:
            result[RIGHT_QUAD_INDEX] = True

    # Draw circles around blobs and display count on screen
    temp_data_with_keypoints = cv2.drawKeypoints(temp_data, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Draw seperator line
    cv2.line(temp_data, (QUAD_SEP, 0), (QUAD_SEP, IMG_HEIGHT * SCALE_FACTOR), (0, 255, 255), 2)

    # Draw count of blobs inside circle and outside circle, as well as the circle itself
    cv2.putText(temp_data_with_keypoints, f"count: {result[RESULT_COUNT_INDEX]}\nleft: {result[LEFT_QUAD_INDEX]}\nright:{result[RIGHT_QUAD_INDEX]}", (10, (IMG_HEIGHT * SCALE_FACTOR) - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("People Counting Subsystem (Thermal) Demo", temp_data_with_keypoints)
    cv2.waitKey(1)

    return(result)

while True:
    print(f"Count:{get_best_of_x(8)}")
