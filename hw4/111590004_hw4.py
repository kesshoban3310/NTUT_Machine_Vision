import os

import cv2
import numpy as np

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
    [ 0,  0,  0],
    [ 1,  2,  1]
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

    # 計算梯度大小（Gradient Magnitude）
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    G = np.clip(G, 0, 255).astype(np.uint8)

    return G


def main():
    input_dir = "test_img/"
    input_file = ["img1", "img2", "img3"]
    output_file = ["_gaussian", "_magnitude", "_result"]
    file_format = ".jpg"
    os.makedirs("result_img", exist_ok=True)
    os.makedirs("report_img", exist_ok=True)

    kernel = generate_kernel()
    for i in range(len(input_file)):
        img_str = input_dir + input_file[i] + file_format
        img = getimg(img_str)
        gray_img = img2grayscale(img)
        gaussian_img = Gaussian_reduction(gray_img, kernel)
        magnitude = Sobel_operation(gaussian_img)

        showimg(input_file[i] + output_file[0], gaussian_img)
        showimg(input_file[i] + output_file[1], magnitude)


if __name__ == '__main__':
    main()
