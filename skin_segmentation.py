import matplotlib.pyplot as mp_plt
import numpy as np
import cv2
from matplotlib.colors import hsv_to_rgb


def bgr_to_hsv(bgr_value):
    bgr = np.uint8([[bgr_value]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    print("%s to HSV is %s" % (str(bgr[0][0]), str(hsv[0][0])))
    return hsv


def hsv_to_bgr(hsv_value):
    hsv = np.uint8([[hsv_value]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("%s to BGR is %s" % (str(hsv[0][0]), str(bgr[0][0])))
    return hsv


def display_color(colors, title=("Lower", "Upper")):
    low_square = np.full((10, 10, 3), colors[0], dtype=np.uint8) / 255.0
    upp_square = np.full((10, 10, 3), colors[1], dtype=np.uint8) / 255.0
    mp_plt.subplot(1, 2, 1)
    mp_plt.title(str(title[0]))
    mp_plt.imshow(hsv_to_rgb(low_square))
    mp_plt.subplot(1, 2, 2)
    mp_plt.title(str(title[1]))
    mp_plt.imshow(hsv_to_rgb(upp_square))
    mp_plt.show()


def h_grid_search():
    range_boundaries_list = []
    for i in range(0, 180, 10):
        for j in range(i + 10, 265, 10):
            hsv_range = [(i, 0, 0), (j, 255, 255)]
            range_boundaries_list.append(hsv_range)
            display_color(hsv_range, hsv_range)

    # best hue range is 0-20
    return range_boundaries_list


def s_grid_search():
    range_boundaries_list = []
    for i in range(0, 265, 10):
        for j in range(i + 10, 265, 10):
            hsv_range = [(0, i, 80), (20, j, 255)]
            range_boundaries_list.append(hsv_range)
            display_color(hsv_range, hsv_range)

    return range_boundaries_list


def sv_grid_search(display=True):
    range_boundaries_list = []
    for i in range(5, 265, 10):
        for j in range(i + 10, 265, 10):
            for k in range(5, 265, 10):
                hsv_range = [(0, i, k), (20, j, 255)]
                range_boundaries_list.append(hsv_range)
                if display:
                    display_color(hsv_range, hsv_range)

    return range_boundaries_list


def segmentation_grid_search(img_name):
    img = cv2.imread(img_name)
    # convert to HSV space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    range_boundaries_list = sv_grid_search(False)

    cv2.imshow(img_name, img)

    for range_boundaries in range_boundaries_list:
        # create masks from ranges
        mask = cv2.inRange(hsv_img, range_boundaries[0], range_boundaries[1])
        final_mask = mask

        # apply mask on image
        segmented = cv2.bitwise_and(img, img, mask=final_mask)

        cv2.imshow(img_name + ' -> [segmented] -> %s' % str(range_boundaries), segmented)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def segmentation(img_name):
    img = cv2.imread(img_name)
    # convert to HSV space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # set lower and upper (H, S, V) boundaries to filter the pixels
    range_boundaries = [(0, 25, 80), (20, 175, 255)]
    # lower range for weaker colors
    range_boundaries1 = [(0, 5, 175), (15, 40, 255)]
    # display_color(range_boundaries)
    # display_color(range_boundaries1)

    # create masks from ranges
    mask = cv2.inRange(hsv_img, range_boundaries[0], range_boundaries[1])
    mask1 = cv2.inRange(hsv_img, range_boundaries1[0], range_boundaries1[1])
    final_mask = mask + mask1

    # apply mask on image
    segmented = cv2.bitwise_and(img, img, mask=final_mask)

    # remove noise
    segmented = cv2.GaussianBlur(segmented, (11, 11), 0)

    # dilate and erode
    kernel = np.ones((9, 9), np.uint8)
    segmented = cv2.dilate(segmented, kernel, iterations=3)
    segmented = cv2.erode(segmented, kernel, iterations=3)

    cv2.imshow(img_name, img)
    cv2.imshow(img_name + ' -> [segmented]', segmented)


def main():
    images = [str(i) + ".jpg" for i in range(1, 11)]

    # run segmentation on every image
    for img in images:
        segmentation(img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # watch output colors by varying ranges
    # h_grid_search()
    # s_grid_search()
    # sv_grid_search()

    # watch many outputs by varying ranges
    # segmentation_grid_search("6.jpg")

    # try some colors
    # hsv = bgr_to_hsv([80, 120, 120])
    # hsv_to_bgr(hsv[0][0])


if __name__ == '__main__':
    main()
