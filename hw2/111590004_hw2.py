import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

matplotlib.use('TKAgg')

four_mask = [[-1, 0], [0, -1]]

eight_mask = [[-1, 0], [0, -1], [-1, -1]]


def np2img(nparr, filename, folder='result_img'):  # Save numpy array to img
    path = os.path.join(folder, filename)
    cv2.imwrite(path, nparr)


def gen_palette():
    ans = np.random.randint(0, 256, (31, 3), dtype=np.uint8)
    ans = np.vstack(([0, 0, 0], ans))
    return ans


def showimg(window_name, img):  # Show Img
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getimg(img_path):
    return cv2.imread(img_path)


def img2array(img):
    return np.array(img, dtype=np.int8)


def img2grayscale(img):
    # np_arr = img[:, :, ::-1]
    gray_arr = 0.11 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.3 * img[:, :, 2]  # 2 -> B, 1 -> G, 0 -> R
    gray_arr = gray_arr.astype(np.uint8)
    return gray_arr


def gray2bin(img, threshold=128):  # problem 1-2
    bin_arr = (img < threshold).astype(np.uint8) * 255
    return bin_arr


def gray2his(img):
    his = [0] * 256
    for i in img:
        for j in i:
            his[j] += 1
    plt.figure(figsize=(8, 4))
    plt.title("Histogram of Gray Image (using for-loop)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.bar(range(256), his, width=1, color='gray')
    plt.grid(True)
    plt.show()


def imgexpand(img, time=1):
    if (time == 0):
        return img
    n, m = len(img), len(img[0])
    ans = np.zeros((n, m), dtype=np.uint8)
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    for i in range(n):
        for j in range(m):
            if (img[i][j] != 0):
                ans[i, j] = 255
                continue
            else:
                IsOne = False
                for k in range(4):
                    nx, ny = i + dx[k], j + dy[k]
                    if (nx < 0 or ny < 0 or nx >= n or ny >= m):
                        continue
                    if (img[nx][ny] != 0):
                        IsOne = True
                if (IsOne):
                    ans[i][j] = 255
                    continue
    ans = imgexpand(ans, time - 1)
    return ans


def imgshrink(img, time=1):
    if (time == 0):
        return img
    n, m = len(img), len(img[0])
    ans = np.zeros((n, m), dtype=np.uint8)
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    for i in range(n):
        for j in range(m):
            if (img[i][j] == 0):
                continue
            else:
                IsZero = False
                for k in range(4):
                    nx, ny = i + dx[k], j + dy[k]
                    if (nx < 0 or ny < 0 or nx >= n or ny >= m):
                        continue
                    if (img[nx][ny] == 0):
                        IsZero = True
                if (not IsZero):
                    ans[i][j] = 255
    ans = imgshrink(ans, time - 1)
    return ans


def imgdraw(img, palette):
    n, m = len(img), len(img[0])
    img = img % 32
    ans = palette[img].astype(np.uint8)

    return ans


def find(x, parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Path compression
        x = parent[x]
    return x


def union(x, y, parent):
    x_root = find(x, parent)
    y_root = find(y, parent)
    if x_root != y_root:
        # Always keep the smaller label
        parent[max(x_root, y_root)] = min(x_root, y_root)


def imglabel(img, mask):
    img = img.astype(np.uint16)
    label = 1
    n, m = img.shape
    parent = [i for i in range(n * m + 1)]  # 初始化 Union-Find 陣列

    for i in range(n):
        for j in range(m):
            if img[i][j] == 0:
                continue
            neighbor = []
            for dx, dy in mask:
                nx, ny = i + dx, j + dy
                if 0 <= nx < n and 0 <= ny < m and img[nx][ny] != 0:
                    neighbor.append(img[nx][ny])
            if len(neighbor) == 0:
                img[i][j] = label
                label += 1
            else:
                min_label = min(neighbor)
                img[i][j] = min_label
                for nb in neighbor:
                    if nb != min_label:
                        union(min_label, nb, parent)

    # 第二遍：將所有 pixel 的 label 縮減為其代表
    label_map = {}
    new_label = 1
    for i in range(n):
        for j in range(m):
            if img[i][j] != 0:
                root = find(img[i][j], parent)
                if root not in label_map:
                    label_map[root] = new_label
                    new_label += 1
                img[i][j] = label_map[root]

    print(new_label - 1)
    return img


def Objcounting(img):
    externel_point = [
        [[0, 0], [0, 255]],
        [[0, 0], [255, 0]],
        [[0, 255], [0, 0]],
        [[255, 0], [0, 0]],
    ]
    internel_point = [
        [[255, 255], [255, 0]],
        [[255, 255], [0, 255]],
        [[255, 0], [255, 255]],
        [[0, 255], [255, 255]],
    ]
    externel, internel = 0, 0
    n, m = len(img), len(img[0])
    for i in range(n - 1):
        for j in range(m - 1):
            a = [[img[i][j], img[i][j + 1]], [img[i + 1][j], img[i + 1][j + 1]]]
            for k in externel_point:
                if(a == k):
                    externel += 1
                    break
            for k in internel_point:
                if(a == k):
                    internel += 1
                    break
    return externel, internel

def main():
    input_dir = "test_img/"
    input_file = ["img1", "img2", "img3"]
    threshold = [227, 254, 254]
    expand_time = [1, 2, 2]
    shrink_time = [1, 1, 1]
    expand_first = [True, False, False]
    output_file = ["_4", "_8"]
    color_palette = gen_palette()
    file_format = ".jpg"
    os.makedirs("result_img", exist_ok=True)
    for i in range(len(input_file)):
        print(f"Processing file: {input_file[i]+file_format}")
        img_str = input_dir + input_file[i] + file_format
        img = getimg(img_str)
        np_img = img2array(img)
        gray_img = img2grayscale(np_img)
        bin_img = gray2bin(gray_img, threshold[i])
        if (expand_first[i]):
            process_img = imgshrink(bin_img, shrink_time[i])
            process_img = imgexpand(process_img, expand_time[i])
        else:
            process_img = imgshrink(bin_img, shrink_time[i])
            process_img = imgexpand(process_img, expand_time[i])

        print("Labeling Using Process_img by 4-neighbor mask: ", end="")
        label_img_4 = imglabel(process_img[:], four_mask)
        color_img_4 = imgdraw(label_img_4, color_palette)
        showimg(input_file[i], color_img_4)

        print("Labeling Using Process_img by 8-neighbor mask: ", end="")
        label_img_8 = imglabel(process_img[:], eight_mask)
        color_img_8 = imgdraw(label_img_8, color_palette)
        showimg(input_file[i], color_img_8)


        externel, internel = Objcounting(bin_img)
        print(f"Using external and internal node method.")
        print(f"External node: {externel}")
        print(f"Internal node: {internel}")
        print(f"Total cound: {(externel - internel)/4}")
        showimg(input_file[i], process_img)

        np2img(color_img_4, input_file[i] + output_file[0] + file_format)
        np2img(color_img_8, input_file[i] + output_file[1] + file_format)
        print(f"Finish process file: {input_file[i]+file_format}")
        print("")


if __name__ == "__main__":
    main()
