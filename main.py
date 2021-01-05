import MLX90640 as mlx
import numpy as np
import cv2

IMG_WIDTH, IMG_HEIGHT = 24, 32
TEMP_MIN, TEMP_MAX = 6, 20

DEPTH_SCALE_FACTOR = 255.0 / (TEMP_MAX - TEMP_MIN)
DEPTH_SCALE_BETA_FACTOR = -TEMP_MIN * 255.0 / (TEMP_MAX - TEMP_MIN)

# Sources:
# https://answers.opencv.org/question/210645/detection-of-people-from-above-with-thermal-camera/
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/

temp_data = np.empty([2, 2])

def counter() -> None:
    mlx.setup(30)
    f = mlx.get_frame()
    mlx.cleanup()
    v_min, v_max = min(f), max(f)

    for x in range(IMG_WIDTH):
        for y in range(IMG_HEIGHT):
            temp_data[x, y] = f[32 * (23 - x) + y]

    # Image processing
    temp_data = temp_data * DEPTH_SCALE_FACTOR + DEPTH_SCALE_BETA_FACTOR
    temp_data[temp_data > 255] = 255
    temp_data[temp_data < 0] = 0
    temp_data = temp_data.astype('uint8')

    temp_data = cv2.resize(temp_data, dsize=(IMG_WIDTH * 10, IMG_HEIGHT * 10), interpolation=cv2.INTER_CUBIC)
    temp_data = cv2.normalize(temp_data, temp_data, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    temp_data = cv2.bilateralFilter(temp_data, 9, 150, 150)
    _, temp_data = cv2.threshold(temp_data, 210, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)

    temp_data = cv2.erode(temp_data, kernel, iterations = 1)
    temp_data = cv2.dilate(temp_data, kernel, iterations = 1)

    temp_data = cv2.morphologyEx(temp_data, cv2.MORPH_CLOSE, kernel)

    temp_data = cv2.bitwise_not(temp_data)

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 7000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.01

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(temp_data)

    # Draw circles around blobs and display count on screen
    temp_data_with_keypoints = cv2.drawKeypoints(temp_data, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(temp_data_with_keypoints, str(len(keypoints)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),3)

    cv2.imshow(temp_data_with_keypoints)

    cv2.waitKey('q')

if __name__ == '__main__':
    while True:
        counter()
