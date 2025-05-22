import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("TkAgg")  # 或 "QtAgg"、"Agg"…

import matplotlib.pyplot as plt
from collections import deque

'''
Template for Machine Vision class.
Include image to nparray, showing image and covert nparray to image.
'''


def np2img(nparr, filename, folder='result_img'):  # Save numpy array to img
    path = os.path.join(folder, filename)
    cv2.imwrite(path, nparr)


def showimg(window_name, img):  # Show Img
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getimg(img_path):  # Get Image from path
    return cv2.imread(img_path)


def img2array(img):  # Convert Image to nparray
    return np.array(img, dtype=np.int8)


def img2grayscale(img):
    # np_arr = img[:, :, ::-1]
    gray_arr = 0.11 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.3 * img[:, :, 2]  # 2 -> B, 1 -> G, 0 -> R
    gray_arr = gray_arr.astype(np.uint8)
    return gray_arr


'''
Main Function to solve HW
'''


def generate_kernel(size=3, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel * (1 / (2 * np.pi * sigma ** 2))


def Gaussian_reduction(img, kernel):
    h, w = img.shape
    padded = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    ans = np.zeros_like(img, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            total = 0.0
            for k in range(3):
                for l in range(3):
                    total += kernel[k, l] * padded[i + k, j + l]
            ans[i, j] = np.clip(total, 0, 255)
    return ans


Sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.int32)

Sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.int32)


def Sobel_operation(img):
    h, w = img.shape
    padded = np.pad(img, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    Gx = np.zeros_like(img, dtype=np.float32)
    Gy = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            x_total = 0.0
            y_total = 0.0
            for k in range(3):
                for l in range(3):
                    x_total += Sobel_x[k, l] * padded[i + k, j + l]
                    y_total += Sobel_y[k, l] * padded[i + k, j + l]
            Gx[i, j] = x_total
            Gy[i, j] = y_total

    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G = np.clip(G, 0, 255).astype(np.uint8)
    Theta = np.arctan2(Gy, Gx)

    return G, Theta


def nms(g, theta):
    h, w = theta.shape
    padded = np.pad(g, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    angle = (np.rad2deg(theta) + 180) % 180
    direction = np.zeros_like(angle, dtype=np.uint8)
    direction[(angle >= 22.5) & (angle < 67.5)] = 1  # 45°
    direction[(angle >= 67.5) & (angle < 112.5)] = 2  # 90°
    direction[(angle >= 112.5) & (angle < 157.5)] = 3  # 135°

    ans = np.zeros_like(g, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            d = direction[i, j]
            center = padded[i + 1, j + 1]

            # 取得方向上的兩個鄰居
            if d == 0:  # 0°
                p1 = padded[i + 1, j]
                p2 = padded[i + 1, j + 2]
            elif d == 1:  # 45°
                p1 = padded[i, j + 2]
                p2 = padded[i + 2, j]
            elif d == 2:  # 90°
                p1 = padded[i, j + 1]
                p2 = padded[i + 2, j + 1]
            elif d == 3:  # 135°
                p1 = padded[i, j]
                p2 = padded[i + 2, j + 2]

            # 比大小
            if center >= p1 and center >= p2:
                ans[i, j] = center
            else:
                ans[i, j] = 0

    return ans.astype(np.uint8)


def gray2his(img):
    his = [0] * 256
    for i in img:
        for j in i:
            '''
            Filter out all zero points,  since they don’t affect the thresholding.
            '''
            if (j == 0):
                continue
            his[j] += 1
    plt.figure(figsize=(8, 4))
    plt.title("Histogram of Gray Image (using for-loop)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.bar(range(256), his, width=1, color='gray')
    plt.grid(True)
    plt.show()


def double_threshold(img, threshold=(4, 10)):
    low, high = threshold
    H, W = img.shape

    result = np.zeros((H, W), dtype=np.uint8)

    strong = img >= high
    weak = (img >= low) & (img < high)

    result[strong] = 255
    result[weak] = 128  # 先標記弱邊緣

    q = deque(zip(*np.nonzero(strong)))

    # 定義 8 個方向
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]

    while q:
        y, x = q.pop()
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if result[ny, nx] == 128:  # 是弱邊緣
                    result[ny, nx] = 255  # 升級成強邊緣

    # 把剩下還是 128 的弱邊緣清掉
    result[result != 255] = 0

    return result


def main():
    input_dir = "test_img/"
    input_file = ["img1", "img2", "img3"]
    output_file = ["_gaussian", "_magnitude", "_result"]
    thresholds = [(6, 15), (30, 50), (5, 10)]
    file_format = ".jpg"
    os.makedirs("result_img", exist_ok=True)
    os.makedirs("report_img", exist_ok=True)

    kernel = generate_kernel()
    for i in range(len(input_file)):
        img_str = input_dir + input_file[i] + file_format
        img = getimg(img_str)
        gray_img = img2grayscale(img)
        gaussian_img = Gaussian_reduction(gray_img, kernel)
        magnitude, theta = Sobel_operation(gaussian_img)
        nms_img = nms(magnitude, theta)

        gray2his(nms_img)
        canny_img = double_threshold(nms_img, thresholds[i])
        showimg(input_file[i] + output_file[0], gaussian_img)
        showimg(input_file[i] + output_file[1], magnitude)
        showimg(input_file[i] + output_file[1], theta)
        showimg(input_file[i] + "_nms", nms_img)
        showimg(input_file[i] + output_file[2], canny_img)

        np2img(gaussian_img, input_file[i] + output_file[0] + file_format)
        np2img(magnitude, input_file[i] + output_file[1] + file_format)
        np2img(canny_img, input_file[i] + output_file[2] + file_format)


if __name__ == '__main__':
    main()
