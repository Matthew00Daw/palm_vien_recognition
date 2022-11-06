import os
import cv2


def crop(img):
    new_img = img[0:500, 0:500]
    cv2.imshow('new', new_img)
    return new_img


if __name__ == '__main__':
    path = 'C:\\Users\\imotw\\PycharmProjects\\diplom_matvey\\ghbv\\'
    imgs = os.listdir(path)
    print(imgs)
    for img in imgs:
        ret = cv2.imread(path+img)
        new_ret = crop(ret)
        cv2.imwrite(path + 'crop' + img, new_ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
