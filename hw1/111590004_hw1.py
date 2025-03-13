import numpy as np
import cv2
import os

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


    color_plate = [(int(i), int(i), int(i)) for i in np.linspace(0, 255, 32)]
    print(color_plate)

    np2img(gray_img1,'img1_q1-1.jpg' , folder)
    np2img(gray_img2,'img2_q1-1 .jpg' , folder)
    np2img(gray_img3,'img3_q1-1.jpg' , folder)
    np2img(bin_img1, 'img1_q1-2.jpg', folder)
    np2img(bin_img2, 'img2_q1-2.jpg', folder)
    np2img(bin_img3, 'img3_q1-2.jpg', folder)


if __name__ == "__main__":
    main()