from multiprocessing.pool import ThreadPool
import numpy as np
import cv2
import os


class GaborFiltering:

    @staticmethod
    def build_filters(ksize: int,
                      sigma: float,
                      lambd: float,
                      gamma: float,
                      psi: float):
        filters = []
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel((ksize, ksize),
                                      sigma,
                                      theta,
                                      lambd,
                                      gamma,
                                      psi,
                                      ktype=cv2.CV_32F)
            kern /= 5 * kern.sum()
            filters.append(kern)

        return filters

    @staticmethod
    def process(img, filters):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)

        return accum

    @staticmethod
    def process_threaded(img, filters, threadn=8):
        accum = np.zeros_like(img)

        def f(kern):
            return cv2.filter2D(img, cv2.CV_8UC3, kern)

        pool = ThreadPool(processes=threadn)
        for fimg in pool.imap_unordered(f, filters):
            np.maximum(accum, fimg, accum)

        return accum

    @staticmethod
    def get_filter_image(image,
                         ksize: int,
                         sigma: float,
                         lambd: float,
                         gamma: float,
                         psi: float):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(src=image, ksize=11)
        nimg2 = cv2.bitwise_not(median)

        return GaborFiltering.process_threaded(img=nimg2,
                                               filters=GaborFiltering.
                                               build_filters(ksize=ksize,
                                                             sigma=sigma,
                                                             lambd=lambd,
                                                             gamma=gamma,
                                                             psi=psi))


class AdditionalFitering:

    @staticmethod
    def inversion(img):
        imgInfo = img.shape
        height = imgInfo[0]
        width = imgInfo[1]

        dst = np.zeros((height, width, 1),
                       np.uint8)

        for i in range(height):
            for j in range(width):
                grayPixel = img[i, j]
                dst[i, j] = 100 - grayPixel

        return dst

    @staticmethod
    def gauss(image):
        return cv2.GaussianBlur(image, (15, 15), 4)

    @staticmethod
    def median(image):
        return cv2.medianBlur(image, 11)

    @staticmethod
    def inversion_1(image):
        return cv2.bitwise_not(image)

    @staticmethod
    def masking(img):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.circle(mask, (210, 210), 200, 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)

        return masked

    @staticmethod
    def skeletonization(img):
        _ = img.copy()
        img = np.array(img)
        size = np.size(img)

        skel = np.zeros(img.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        done = False

        while not done:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True
        return skel

    @staticmethod
    def get_final_picture(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filter_image = AdditionalFitering.median(gray_image)
        invers_image = AdditionalFitering.inversion_1(filter_image)
        first = GaborFiltering.process_threaded(invers_image,
                                                GaborFiltering.
                                                build_filters(ksize=20,
                                                              sigma=20.2,
                                                              lambd=19.5,
                                                              gamma=2.1,
                                                              psi=0))
        cv2.imwrite('first180.png', first)
        final_img = np.zeros([420, 420], np.uint8)
        res = cv2.inRange(first, 150, 255)
        ret, thresh = cv2.threshold(res, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_TC89_L1)
        first_cont = cv2.drawContours(final_img, contours, -1,
                                      (255, 255, 255), -1)
        cv2.imwrite('first_cont180.png', first_cont)

        second = GaborFiltering.process_threaded(first_cont,
                                                 GaborFiltering.
                                                 build_filters(ksize=23,
                                                               sigma=40,
                                                               lambd=24.5,
                                                               gamma=5.9,
                                                               psi=0))
        cv2.imwrite('second180.png', second)

        final_img = np.zeros([420, 420], np.uint8)
        res = cv2.inRange(second, 150, 255)
        ret, thresh = cv2.threshold(res, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_TC89_L1)
        first_cont = cv2.drawContours(final_img, contours, -1,
                                      (255, 255, 255), -1)
        return AdditionalFitering.masking(AdditionalFitering.skeletonization(first_cont))


if __name__ == '__main__':
    path = 'test_db/'
    imgs = os.listdir(path)
    for img in imgs:
        image = cv2.imread(path + img)
        res = AdditionalFitering.get_final_picture(image=image)
        cv2.imwrite('C:\\Users\\imotw\\PycharmProjects\\diplom_matvey\\skel_db\\' + img, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # image = cv2.imread('180.bmp')
    # res = AdditionalFitering.get_final_picture(image=image)
    #
    # # cv2.imshow('sds', first_cont)
    # cv2.imwrite('skel_180.png', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
