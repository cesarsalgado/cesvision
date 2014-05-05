import cv2
import matplotlib.pyplot as plt

def show_image(img, name='image'):
  cv2.imshow(name, img)
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
