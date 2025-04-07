import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

four_mask = [[-1,0],[0,-1]]

eight_mask = [[-1,0],[0,-1],[-1,-1]]
def showimg(window_name,img): #Show Img
    cv2.imshow(window_name,img)
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
def gray2bin(img,threshold = 128): # problem 1-2
    bin_arr = (img < threshold).astype(np.uint8) * 255
    return bin_arr

def gray2his(img):
    his = [0]*256
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

def imglabel(img,mask):
    label = 1
    n,m = len(img),len(img[0])

    for i in range(n):
        for j in range(m):
            if(img[i][j]==0):
                continue
            neighbor = []
            for k in mask:
                nx,ny = i+k[0],j+k[1]
                if(nx<0 or ny<0 or nx>=n or ny>=m):
                    continue
                if(img[nx][ny]==0):
                    continue
                neighbor.append(img[nx][ny])
            if(len(neighbor) == 0):
                label+=1
                img[i][j] = label
            elif(len(neighbor) == 1):
                img[i][j] = neighbor[0]
            else:
                is_same = True
                for k in range(1,len(neighbor)):
                    if(neighbor[k] != neighbor[k-1]):
                        is_same = False
                        break
                if(is_same):
                    img[i][j] = neighbor[0]
                else:
                    img[i][j] = neighbor[len(neighbor)-1]
    return img


def main():
    input_dir = "test_img/"
    output_dir = "result_img/"
    input_file = ["img1","img2","img3"]
    threshold = [227,254,254]
    output_file = ["_4","_8"]
    file_format = ".jpg"
    os.makedirs("result_img", exist_ok=True)
    for i in range(len(input_file)):
        img_str = input_dir + input_file[i] + file_format
        img = getimg(img_str)
        np_img = img2array(img)
        gray_img = img2grayscale(np_img)
        bin_img = gray2bin(gray_img,threshold[i])
        # label_img_4 = imglabel(bin_img[:],four_mask)
        showimg(input_file[i],bin_img)


if __name__ == "__main__":
    main()