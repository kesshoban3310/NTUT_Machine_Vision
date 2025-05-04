import heapq
import os
import cv2
import numpy as np

default_colors = [  # Colors for image labeling
    (0, 0, 0),  # 0: Unlabeled (Black)
    (0, 0, 255),  # 1: Red
    (0, 255, 0),  # 2: Green
    (255, 0, 0),  # 3: Blue
    (255, 255, 0),  # 4: Cyan / Light Blue
    (255, 0, 255),  # 5: Magenta / Purple
    (0, 255, 255),  # 6: Yellow
    (128, 128, 0),  # 7: Olive
    (0, 128, 128),  # 8: Teal
    (128, 0, 128)  # 9: Violet / Dark Magenta
]
label_colors = {i: default_colors[i] for i in range(10)}
label_colors[-1] = (128, 128, 128)
label_colors[-2] = (255, 255, 255)

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


def mouse_marked(img, labels=9):
    img_copy = img.copy()
    h, w = img.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)

    current_label = 1  # start at label 1

    # Progress muse event
    def mouse_callback(event, x, y, flags, param):
        nonlocal img_copy, label_map, current_label
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(img_copy, (x, y), 5, label_colors[current_label], -1)
            cv2.circle(label_map, (x, y), 5, current_label, -1)

    cv2.namedWindow("Mouse Mark Tool")
    cv2.setMouseCallback("Mouse Mark Tool", mouse_callback)

    print("Use the mouse to mark regions on the image:")
    print("  Press keys 1~9 to select labels, press 'q' or click the X to exit")

    while True:
        if cv2.getWindowProperty("Mouse Mark Tool", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed, exiting the labeling tool automatically")
            break

        cv2.imshow("Mouse Mark Tool", img_copy)
        key = cv2.waitKey(1) & 0xFF

        if ord('1') <= key <= ord('9'):
            label_num = key - ord('0')
            if label_num <= labels:
                current_label = label_num
                print(f"Switched to label {current_label}")
        elif key == ord('q'):
            print("Pressed 'q', exiting the labeling tool")
            break

    cv2.destroyAllWindows()
    return label_map, img_copy


def label2maskimg(label_img, ori_img):
    h, w = ori_img.shape[:2]
    ans = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if np.array_equal(label_img[i][j], ori_img[i][j]):
                ans[i][j] = (0, 0, 0)
            else:
                ans[i][j] = label_img[i][j]
    return ans


four_neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def watershedsegment(img, label_map):
    h, w = img.shape[:2]

    gray = img2grayscale(img)

    ans = label_map.astype(np.int32)
    pq = []
    for y in range(h):
        for x in range(w):
            if ans[y, x] > 0:
                for dy, dx in four_neigh:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and ans[ny, nx] == 0:
                        ans[ny, nx] = -2
                        heapq.heappush(pq, (gray[ny, nx], ny, nx))

    while pq:
        _, y, x = heapq.heappop(pq)
        neighbor_labels = set()

        for dy, dx in four_neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                lbl = ans[ny, nx]
                if lbl > 0:
                    neighbor_labels.add(lbl)

        if len(neighbor_labels) == 1:
            ans[y, x] = neighbor_labels.pop()
        elif len(neighbor_labels) >= 2:
            ans[y, x] = -1  # Edge

        for dy, dx in four_neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and ans[ny, nx] == 0:
                ans[ny, nx] = -2
                heapq.heappush(pq, (gray[ny, nx], ny, nx))

    return ans


def watershedtoimg(label_img, ori_img):
    h, w = label_img.shape[:2]
    ans = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            if(label_img[i][j] == -1 or label_img[i][j] == -2):
                ans[i, j] = label_colors[label_img[i, j]]
                continue

            pixel = ori_img[i, j].astype(np.int32)
            label_val = list(label_colors[label_img[i, j]])
            pixel += label_val
            pixel = np.clip(pixel, 0, 255)
            ans[i, j] = pixel.astype(np.uint8)

    return ans


def main():
    input_dir = "test_img/"
    input_file = ["img1", "img2", "img3"]
    output_file = ["_marked", "_mask", "_seg"]
    file_format = ".png"
    output_format = ".jpg"
    os.makedirs("result_img", exist_ok=True)
    os.makedirs("report_img", exist_ok=True)
    for i in range(len(input_file)):
        img_str = input_dir + input_file[i] + file_format
        img = getimg(img_str)
        label_map, mark_img = mouse_marked(img)  # Get Labeling Img
        mask_img = label2maskimg(mark_img, img)  # Get Mask

        watershedarr = watershedsegment(img, label_map)

        watershedimg = watershedtoimg(watershedarr,img)

        showimg("Test", watershedimg)

        np2img(img2grayscale(img), input_file[i] + "_grayscale" + output_format, 'report_img')

        np2img(mark_img, input_file[i] + output_file[0] + output_format)
        np2img(mask_img, input_file[i] + output_file[1] + output_format)
        np2img(watershedimg, input_file[i] + output_file[2] + output_format)


if __name__ == '__main__':
    main()