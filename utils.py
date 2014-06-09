import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image(img, name='image'):
  img_to_show = img.astype(np.uint8) if img.dtype != np.uint8 else img
  cv2.imshow(name, img_to_show)
  cv2.waitKey(-1)

def show_image2(img):
  if len(img.shape) == 2:
    plt.imshow(img, cmap=plt.cm.gray, interpolation='none')
  else:
    plt.imshow(img, interpolation='none')
  plt.show()

def dall():
  cv2.destroyAllWindows()

def to_gray(im):
  return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def load_img(path, gray=False):
  img = cv2.imread(path)
  if gray:
    img = to_gray(img)
  return img
