import numpy as np
import cv2
import os
from collections import Counter
from scipy.spatial import KDTree
def np2img(nparr,filename,folder): # Save numpy array to img
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
    bin_arr = np.where(img >= threshold,255,0).astype(np.uint8)
    return bin_arr
def img2idximg(img): # problem 1-3
    h, w, _ = img.shape
    pixel = img.reshape(-1, 3)

    # 將 RGB 轉 tuple
    pixel_tuples = [tuple(rgb) for rgb in pixel]
    color_count = Counter(pixel_tuples)
    adv_32 = []
    fre_color = color_count.most_common()
    gz = len(fre_color) // 32

    # 取出現頻率最高跟最低，共32種顏色
    for i in range(32):
        start = i * gz
        end = (i + 1) * gz if i != 31 else len(fre_color)  # 最後一群全包
        group = fre_color[start:end]
        group_colors = np.array([color[0] for color in group])  # 只取顏色 (不取頻率)
        avg_color = np.mean(group_colors, axis=0)  # 計算每群顏色的均值
        adv_32.append(avg_color)

    color_tree = KDTree(adv_32)
    _,indices = color_tree.query(pixel)
    adv_32_array = np.array(adv_32)


    ans = adv_32_array[indices].reshape(h, w, 3).astype(np.uint8)

    return ans



def main():
    img1 = cv2.imread('test_img/img1.jpg')
    img2 = cv2.imread('test_img/img2.jpg')
    img3 = cv2.imread('test_img/img3.jpg')
    folder = 'result_img'

    gray_img1 = rgb2gray(img1)
    gray_img2 = rgb2gray(img2)
    gray_img3 = rgb2gray(img3)

    bin_img1 = gray2bin(gray_img1)
    bin_img2 = gray2bin(gray_img2)
    bin_img3 = gray2bin(gray_img3)
    idx_img1 = img2idximg(img1)
    idx_img2 = img2idximg(img2)
    idx_img3 = img2idximg(img3)

    np2img(gray_img1,'img1_q1-1.jpg' , folder)
    np2img(gray_img2,'img2_q1-1.jpg' , folder)
    np2img(gray_img3,'img3_q1-1.jpg' , folder)
    np2img(bin_img1, 'img1_q1-2.jpg', folder)
    np2img(bin_img2, 'img2_q1-2.jpg', folder)
    np2img(bin_img3, 'img3_q1-2.jpg', folder)
    np2img(idx_img1, 'img1_q1-3.jpg', folder)
    np2img(idx_img2, 'img2_q1-3.jpg', folder)
    np2img(idx_img3, 'img3_q1-3.jpg', folder)


if __name__ == "__main__":
    main()