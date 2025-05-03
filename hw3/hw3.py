import cv2
import numpy as np
import os



def mouse_marked(event, x, y, flags, param):
    return 0









def np2img(nparr, filename, folder='result_img'):  # Save numpy array to img
    path = os.path.join(folder, filename)
    cv2.imwrite(path, nparr)

def showimg(window_name, img):  # Show Img
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getimg(img_path): # Get Image from path
    return cv2.imread(img_path)


def img2array(img): # Convert Image to nparray
    return np.array(img, dtype=np.int8)


def main():
    input_dir = "test_img/"
    input_file = ["img1", "img2", "img3"]
    output_file = ["_marked", "_mask", "_seg"]
    file_format = ".png"
    output_format = ".jpg"
    os.makedirs("result_img", exist_ok=True)
    for i in range(len(input_file)):
        img_str = input_dir + input_file[i] + file_format






if __name__ == '__main__':
    main()