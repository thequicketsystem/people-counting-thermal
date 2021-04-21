import board
import busio
import adafruit_mlx90640
import numpy as np
import cv2
from math import sqrt
from concurrent.futures import ThreadPoolExecutor

# Sources:
# https://answers.opencv.org/question/210645/detection-of-people-from-above-with-thermal-camera/
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/
# https://github.com/thequicketsystem/people-counting-visual/
# https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

IMG_WIDTH, IMG_HEIGHT = 32, 24
TEMP_MIN, TEMP_MAX = 6, 20

SCALE_FACTOR = 10

SCALED_WIDTH, SCALED_HEIGHT = IMG_WIDTH * SCALE_FACTOR, IMG_HEIGHT * SCALE_FACTOR

MIN_TEMP = 30

SEP = (IMG_WIDTH * SCALE_FACTOR) // 2

POLLING_FRAMES_LENGTH = 4
EXTENSION_LENGTH = 8

CONFIDENCE_THRESHOLD = 0

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
params.minArea = 700
params.maxArea = 7000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detectors = [cv2.SimpleBlobDetector_create(params) for i in range(2)]

# TODO: Major cleanup/un-spaghettification needed if this does actually work
def get_frame_data() -> int:

    frames = POLLING_FRAMES_LENGTH
    is_ext = False

    top_data, bottom_data = 0, 0

    while frames > 0 and not (bottom_data > CONFIDENCE_THRESHOLD and top_data > CONFIDENCE_THRESHOLD): 
        try:
            mlx.getFrame(f)
        except ValueError:
            pass
        
        temp_data = np.array(f).reshape((IMG_HEIGHT, IMG_WIDTH))

        temp_data = cv2.resize(temp_data, dsize=(SCALED_WIDTH, SCALED_HEIGHT))
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

        temp_data_top, temp_data_bottom = temp_data[:,:SCALED_HEIGHT], temp_data[:,SCALED_HEIGHT:]
        
        keypoints = []

        # process the two halves in seperate threads
        # this will need to be cleaned up a lot later. no magic numbers!
        with ThreadPoolExecutor() as ex:
            td_future = ex.submit(detectors[0].detect, temp_data_top)
            bd_future = ex.submit(detectors[1].detect, temp_data_bottom)

            # join the results together
            keypoints.extend(td_future.result())
            keypoints.extend(bd_future.result())


        if len(keypoints) == 1 and not is_ext:
            frames += EXTENSION_LENGTH
            is_ext = True

        pts = cv2.KeyPoint_convert(keypoints)
        for point in pts:
            if point[1] < SEP:
                top_data += 1
            else:
                bottom_data += 1

        # Draw circles around blobs and display count on screen
        output_frame = cv2.drawKeypoints(temp_data, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Draw count of blobs inside circle and outside circle, as well as the circle itself
        cv2.putText(output_frame, f"top: {top_data}", (10, SCALED_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(output_frame, f"bottom: {bottom_data}", (10, SCALED_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(output_frame, (0, SCALED_HEIGHT // 2), (SCALED_WIDTH, SCALED_HEIGHT // 2), (0, 255, 255), 2)

        cv2.imshow("People Counting Subsystem (Thermal) Demo", output_frame)
        cv2.waitKey(1)

        frames -= 1

    return len([x for x in [top_data, bottom_data] if x > CONFIDENCE_THRESHOLD])


while True:
    print(f"Count:{get_frame_data(16)}")
