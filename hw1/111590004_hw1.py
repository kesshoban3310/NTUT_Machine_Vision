import numpy as np
import cv2
import os
from collections import Counter
from scipy.spatial import KDTree
def np2img(nparr,filename,folder = 'result_img'): # Save numpy array to img
    path = os.path.join(folder,filename)
    cv2.imwrite(path,nparr)

def showimg(window_name,img): #Show Img
    cv2.imshow(window_name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def rgb2gray(img): # problem 1-1
    np_arr = np.array(img, dtype=np.uint8)
    np_arr = np_arr[:,:,::-1]
    gray_arr = 0.11 * np_arr[:,:,2] + 0.59 * np_arr[:,:,1] + 0.3 * np_arr[:,:,0] # 2 -> B, 1 -> G, 0 -> R
    gray_arr = gray_arr.astype(np.uint8)
    return gray_arr

def gray2bin(img,threshold = 128): # problem 1-2
    bin_arr = (img >= threshold).astype(np.uint8) * 255
    return bin_arr
def img2idximg(img): # problem 1-3
    h, w, _ = img.shape
    pixel = img.reshape(-1, 3)

    # 去除重複顏色
    unique_colors = np.unique(pixel, axis=0)
    colors_list = unique_colors.tolist()

    # 進行 Median Cut 分群
    def median_cut(colors, depth):
        if depth == 5 or len(colors) == 0:  # 因為 2^5 = 32
            avg_color = np.mean(colors, axis=0)
            return [avg_color]
        else:
            colors = np.array(colors)
            ranges = np.ptp(colors, axis=0)  # peak-to-peak: max - min
            axis = np.argmax(ranges)  # 找出最大範圍的 channel

            sorted_colors = sorted(colors.tolist(), key=lambda x: x[axis])

            mid = len(sorted_colors) // 2
            left = median_cut(sorted_colors[:mid], depth + 1)
            right = median_cut(sorted_colors[mid:], depth + 1)

            return left + right

    palette = median_cut(colors_list, 0)
    palette = np.array(palette)

    print(len(palette))

    color_tree = KDTree(palette)
    _, indices = color_tree.query(pixel)

    ans = palette[indices].reshape(h, w, 3).astype(np.uint8)
    return ans

def scaleimgwithinterp(img,scale): # problem 2-2
    h, w, _ = img.shape
    nh, nw = int(h * scale), int(w * scale)
    ans = np.zeros((nh,nw, _), dtype=np.uint8)
    for i in range(nh):
        for j in range(nw):
            # 對應到原圖座標
            x = (i + 0.5) / scale - 0.5
            y = (j + 0.5) / scale - 0.5

            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, h - 1)
            y2 = min(y1 + 1, w - 1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            dx = x - x1
            dy = y - y1

            for k in range(_):
                Q11 = img[x1, y1, k]
                Q21 = img[x1, y2, k]
                Q12 = img[x2, y1, k]
                Q22 = img[x2, y2, k]

                R1 = (1 - dy) * Q11 + dy * Q21
                R2 = (1 - dy) * Q12 + dy * Q22
                P = (1 - dx) * R1 + dx * R2

                ans[i, j, k] = np.clip(P, 0, 255)
    return ans

def scaleimg(img,scale): # problem 2-1
    h, w, _ = img.shape
    nh,nw = int(h*scale), int(w*scale)
    ans = np.zeros((nh,nw, 3), dtype=np.uint8)
    for i in range(nh):
        for j in range(nw):
            ans[i, j] = img[int(i / scale), int(j / scale)]
    return ans




def main():
    file = 'test_img/'
    img = ['img1','img2','img3']
    jpg = '.jpg'
    pb = ['_q1-1','_q1-2','_q1-3','_q2-1_half','_q2-1_double','_q2-2_half','_q2-2_double']
    for i in img:
        pic = cv2.imread(file+i+jpg)
        gray_img = rgb2gray(pic)
        bin_img = gray2bin(gray_img)
        idx_img = img2idximg(pic)
        half_img = scaleimg(pic,0.5)
        double_img = scaleimg(pic,2.0)
        half_img_wint = scaleimgwithinterp(pic, 0.5)
        double_img_wint = scaleimgwithinterp(pic, 2.0)

        np2img(gray_img,i + pb[0] + jpg)
        np2img(bin_img, i + pb[1] + jpg)
        np2img(idx_img, i + pb[2] + jpg)
        np2img(half_img, i + pb[3] + jpg)
        np2img(double_img, i + pb[4] + jpg)
        np2img(half_img_wint, i + pb[5] + jpg)
        np2img(double_img_wint, i + pb[6] + jpg)

    print("All Pictures Done!")


if __name__ == "__main__":
    main()