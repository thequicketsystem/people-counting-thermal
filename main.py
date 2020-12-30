import MLX90640 as mlx
import numpy as np
import cv2

IMG_WIDTH, IMG_HEIGHT = 24, 32
TEMP_MIN, TEMP_MAX = 6, 20

DEPTH_SCALE_FACTOR = 255.0 / (TEMP_MAX - TEMP_MIN)
DEPTH_SCALE_BETA_FACTOR = -TEMP_MIN * 255.0 / (TEMP_MAX - TEMP_MIN)

# https://answers.opencv.org/question/210645/detection-of-people-from-above-with-thermal-camera/

temp_data = np.empty([2, 2])

def counter() -> None:
    mlx.setup(30)
    f = mlx.get_frame()
    mlx.cleanup()
    v_min, v_max = min(f), max(f)

    for x in range(IMG_WIDTH):
        for y in range(IMG_HEIGHT):
            temp_data[x, y] = f[32 * (23 - x) + y]

    temp_data = temp_data * DEPTH_SCALE_FACTOR + DEPTH_SCALE_BETA_FACTOR
    temp_data[temp_data > 255] = 255
    temp_data[temp_data < 0] = 0
    temp_data = temp_data.astype('uint8')

    temp_data = cv2.resize(temp_data, dsize=(IMG_WIDTH * 10, IMG_HEIGHT * 10), interpolation=cv2.INTER_CUBIC)
    temp_data = cv2.normalize(temp_data, temp_data, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    