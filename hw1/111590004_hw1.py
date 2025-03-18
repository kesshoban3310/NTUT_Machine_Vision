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

def scaleimgwithinterp(img,scale): # problem 2-1
    h, w, _ = img.shape
    nh, nw = int(h * scale), int(w * scale)
    ans = np.zeros((nh,nw, _), dtype=np.uint8)
    for i in range(nh):
        for j in range(nw):
            x = (i + 0.5) / scale - 0.5
            y = (j + 0.5) / scale - 0.5

            x1 = int(np.floor(x))
            x2 = min(x1 + 1, h - 1)
            y1 = int(np.floor(y))
            y2 = min(y1 + 1, w - 1)

            x1 = max(0, x1)
            y1 = max(0, y1)

            dx1,dx2,dex = x2-x,x - x1,x2-x1
            dy1,dy2,dey = y2-y,y - y1,y2-y1

            for c in range(_):
                Q11 = img[x1, y1, c]
                Q21 = img[x1, y2, c]
                Q12 = img[x2, y1, c]
                Q22 = img[x2, y2, c]
                R1 = (dx1/dex) * Q11 + (dx2/dex) * Q21
                R2 = (dx1/dex) * Q12 + (dx2/dex) * Q22
                P = (dy1/dex) * R1 + (dy2/dex) * R2
                ans[i, j, c] = np.clip(P, 0, 255)
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
    pb = ['_q1-1','_q1-2','_q1-3','_q2-1_half','_q2-1_double','_q2-2_half','q2-2_double']
    for i in img:
        pic = cv2.imread(file+i+jpg)
        gray_img = rgb2gray(pic)
        bin_img = gray2bin(gray_img)
        idx_img = img2idximg(pic)
        half_img = scaleimg(pic,0.5)
        double_img = scaleimg(pic,2.0)
        half_img_wint = scaleimg(pic, 0.5)
        double_img_wint = scaleimg(pic, 2.0)

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