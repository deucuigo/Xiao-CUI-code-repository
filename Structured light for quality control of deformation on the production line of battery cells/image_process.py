import cv2
from skimage import morphology
from scipy.optimize import curve_fit
import numpy as np
import time


def max_point(img, offset_of_extracted_points=5, threshold=200):
    """
    implement the ridge line tracing method(find the point with Max.value in each row)
    :param threshold: the threshold to eliminate the background light, the part with brightness below threshold are set to be zero.
    :param img: the path of the image
    :param offset_of_extracted_points: the extracted point in each n rows
    :return: a series of coordinates of the extracted points
    """
    img = cv2.inRange(img, threshold, 255)  # threshold method
    i_summer = []
    for y in range(0, len(img), offset_of_extracted_points):
        imax = max(img[y])
        i = np.where(img[y] == imax)[0][0]
        if imax > 100:
            i_summer.append([i, y])
    return i_summer


def max_point_fit(img, offset_of_extracted_points=5, threshold=200):
    """
    implement the ridge line tracing method(find the point with Max.value in each row), the using fitting method to make the accuracy of the method.
    :param threshold: the threshold to eliminate the background light, the part with brightness below threshold are set to be zero.
    :param img: the path of the image
    :param offset_of_extracted_points: the extracted point in each n rows
    :return: a series of coordinates of the extracted points
    """

    def func(x, a, u, sig):
        return a * np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (sig * np.sqrt(2 * np.pi))

    img = cv2.inRange(img, threshold, 255)  # threshold method
    i_summer = []
    for y in range(0, len(img), offset_of_extracted_points):
        imax = max(img[y])
        i = np.where(img[y] == imax)[0][0]
        if (imax > 100):
            try:
                popt, __ = curve_fit(func, np.linspace(0, len(img[y]), len(img[y])), img[y])
            except:
                i_summer.append([i, y])
            else:

                if i - 2 <= popt[1] <= i + 2:
                    i_summer.append([popt[1], y])
                else:
                    i_summer.append([i, y])
    return i_summer


def gray(img, offset_of_extracted_points=5, threshold=200):
    """
    implement the barycenter method
    :param img: the path of the image
    :param offset_of_extracted_points: the extracted point in each n rows
    :param threshold: the threshold to eliminate the background light, the part with brightness below threshold are set to be zero.
    :return: a series of coordinates of the extracted points
    """
    img = cv2.inRange(img, offset_of_extracted_points, 255)
    cv2.imshow("gray_show", img)
    i_summer = []
    for y in range(0, len(img), offset):
        n = np.array(np.where(img[y] > 30)[0])
        v = np.array([img[y][i_n] for i_n in n])
        v_c = v * n
        if len(n) > 0:
            pos = sum(v_c) / sum(v)
            i_summer.append([pos, y])
    return i_summer


def ske(img, threshold=200):
    """
    implement skeleton extraction method
    :param img: the path of the image
    :param threshold: the threshold to eliminate the background light, the part with brightness below threshold are set to be zero.
    :return: a series of coordinates of the extracted points
    """
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # get the resulted binary image
    # cv2.imshow("sk_show", binary)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    index = [(i[1], i[0]) for i in np.argwhere(skeleton0 == 1)]  # extracting coordinates
    return index


def get_laser(img, method='max', threshold=200):
    # preprocess -- converse colored images to grayscale image
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # #beginn = time.time() # to test the speed
    # # # offset_of_extracted_points is to adjust the density of the laser point extraction
    # # # Fitting makes the speed slow

    if method == 'max':
        i_sam = max_point(img, 10, threshold, )
    if method == 'max_fitting':
        i_sam = max_point_fit(img, 10, threshold)
    if method == 'gray':
        i_sam = gray(img, 10, threshold)
    if method == 'skeleton':
        i_sam = ske(img, threshold)

    # # get the speed of each method
    # end = time.time()
    # print(f"time: {end - beginn}", )

    # set count point
    img_result = np.zeros(np.shape(img))
    for i in i_sam:
        cv2.circle(im_gray, (int(i[0]), i[1]), 1, (0, 0, 0))
    # show result images
    cv2.imshow("laser", im_gray)

    return np.array(i_sam)


# space -- take pictures; ecs -- finish taking pictures
def get_Picture(pict_name, folder_path):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    scale = 0.6
    num = 0
    # cstr = input("type in the name: ")

    success, img = cap.read()

    while True:
        success, img = cap.read()  # read in the source

        img_ = cv2.resize(img, (960, 540), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Video", img_)  # outpuut the video
        key = cv2.waitKey(1)

        if key == 27:
            break
        if key == 32:
            filename = f"{folder_path}/{pict_name}{str(num)}.png"
            print(f"{pict_name}{str(num)}.png")
            cv2.imwrite(filename, img)
            num += 1
