from multiprocessing.pool import ThreadPool
from PIL import Image
import numpy as np
import cv2


def get_image(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(src=img, ksize=11)
    nimg = cv2.bitwise_not(median)

    return nimg


def build_filters(ksize: int = 21,
                  sigma: float = 21,
                  lambd: float = 21,
                  gamma: float = 2,
                  psi: float = 0):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        kern /= 5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def process_threaded(img, filters, threadn=8):
    accum = np.zeros_like(img)

    def f(kern):
        return cv2.filter2D(img, cv2.CV_8UC3, kern)

    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum


def inversion(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]

    dst = np.zeros((height, width, 1),
                   np.uint8)  # определяет новый холст, параметр «1» указывает, что пиксель состоит из цвета

    for i in range(height):
        for j in range(width):
            grayPixel = img[i, j]  # текущее значение серого
            dst[i, j] = 100 - grayPixel

    return dst


def masking(img):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.circle(mask, (210, 210), 200, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked


def skeletonization(img):
    # img = cv2.imread(path, 0)

    # Threshold the image
    # ret, img = cv2.threshold(img, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    # Step 3: Substract open from the original image
    temp = cv2.subtract(img, open)
    # Step 4: Erode the original image and refine the skeleton
    # eroded = cv2.erode(img, element)
    # skel = cv2.bitwise_or(skel, temp)

    return temp


def background_remoover(img):
    original = img.copy()
    l = int(max(5, 6))
    u = int(min(6, 6))

    # ed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(img, (21, 51), 3)
    # edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(edges, l, u)

    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    data = mask.tolist()
    # sys.setrecursionlimit(10 ** 8)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] != 255:
                data[i][j] = -1
            else:
                break
        for j in range(len(data[i]) - 1, -1, -1):
            if data[i][j] != 255:
                data[i][j] = -1
            else:
                break
    image = np.array(data)
    image[image != -1] = 255
    image[image == -1] = 0

    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask == 0] = 255
    cv2.imwrite('bg.png', result)

    img = Image.open('bg.png')
    img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    return newData

def Canny_filter(image):
    hsv = cv2.inRange(image, cv2.COLOR_BGR2HSV, upperb=150)
    lower_b = np.array([90, 90, 90])
    upper_b = np.array([150, 150, 150])
    mask = cv2.inRange(hsv, lower_b, upper_b)
    res = cv2.bitwise_and(image, image, mask=mask)
    edges = cv2.Canny(image, 100, 200)
    return edges

if __name__ == '__main__':
    img = cv2.imread('122.bmp')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = masking(img)

    fimg = cv2.GaussianBlur(img, (15, 15), 4)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    # lap = np.uint8(np.absolute(lap))

    median = cv2.medianBlur(src=img, ksize=11)

    nimg = cv2.bitwise_not(median)
    nimg2 = cv2.bitwise_not(fimg)

    # res1 = process(nimg2, filters)
    # res2 = process_threaded(nimg2, filters)

    cv2.namedWindow('result')

    cv2.createTrackbar('ksize', 'result', 0, 60, lambda x: None)
    cv2.createTrackbar('sigma', 'result', 0, 400, lambda x: None)
    # cv2.createTrackbar('theta', 'result', 100, 400, )
    cv2.createTrackbar('lambd', 'result', 0, 400,lambda x: None)
    cv2.createTrackbar('gamma', 'result', 0, 60,lambda x: None)
    cv2.createTrackbar('psi', 'result', 0, 100,lambda x: None)
    while (1):
        # Вернуть значение позиции ползунка
        ksize = cv2.getTrackbarPos('ksize', 'result')
        sigma = cv2.getTrackbarPos('sigma', 'result')
        lambd = cv2.getTrackbarPos('lambd', 'result')
        gamma = cv2.getTrackbarPos('gamma', 'result')
        psi = cv2.getTrackbarPos('psi', 'result')

        result = process_threaded(median, build_filters(ksize, sigma/10, lambd/10, gamma/10, psi/10))

        cv2.imshow('blur', median)
        cv2.imshow('result', masking(result))
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.imwrite('f122.bmp', masking(result))

    # print('res1 == res2: ', (res1 == res2).all())
    # res2 = process(res1, filters)
    # res2 = process(res2, filters)
    # res2 = process(res2, filters)
    # result = process_threaded(nimg, build_filters(ksize=23, sigma=40, lambd=24.5, gamma=5.9, psi=0))
    # result = process_threaded(nimg, build_filters(ksize=20, sigma=20.2, lambd=19.5, gamma=2.1, psi=0))
    # result = process_threaded(nimg, build_filters(ksize=17, sigma=18.3, lambd=17.6, gamma=2, psi=0))
    # result = process_threaded(result, build_filters(ksize=13, sigma=5, lambd=9.5, gamma=1.5, psi=0))
    # result = process_threaded(result, build_filters(ksize=13, sigma=5, lambd=9.5, gamma=1.5, psi=0))
    # cv2.imshow('itr', result)

    # th3 = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

    # final_img = np.zeros([420, 420, 4], np.uint8)
    # final_img = np.zeros([240, 320, 4], np.uint8)
    # res = cv2.inRange(result, 150, 255)
    # ret, thresh = cv2.threshold(res, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(final_img, contours, -1, (255, 255, 255), -1)
    #
    # cv2.imshow('original',img)
    # cv2.imshow('res1', final_img)
    #

    # final = cv2.erode(final_img.copy(), None, iterations=2)
    #
    # final2 = skeletonization(final_img)
    # cv2.imshow('erode', final)
    # cv2.imshow('skelet', final2)
    # cv2.imshow('img', img)
    # cv2.imshow('wb', background_remoover(img))
    # cv2.imshow('mask', masking(img))
    # cv2.imshow('median', median)
    # cv2.imshow('lap', lap)
    # cv2.imshow('result', result)
    # cv2.imshow('FIMG', fimg)

    # cv2.imshow('Canny', Canny_filter(img))
    # cv2.imshow('contours', final)
    cv2.waitKey()
    cv2.destroyAllWindows()
