import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from cesarpy.io import get_all_file_names_from_dir

def show_image(img, name='image', waitkey_forever=True):
  img_to_show = img.astype(np.uint8) if img.dtype != np.uint8 else img
  if img.dtype == bool:
    img_to_show *= 255
  cv2.imshow(name, img_to_show)
  if waitkey_forever:
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

def read_images_from_dir(dir_path, imgs_ext, gray=False, sort=True):
  files = cesarpy.io.get_all_file_names_from_dir(dir_path, imgs_ext, sort=sort)
  imgs = []
  for f in files:
    imgs.append(load_img(os.path.join(dir_path, f), gray))
  return imgs

def apply_to_each_channel(img, func):
  if len(img.shape) == 2:
    return func(img)
  ch_list = []
  nch = img.shape[2]
  for i in xrange(nch):
    ch_list.append(func(img[:,:,i]))
  return np.dstack(tuple(ch_list))

def naive_zoom(image, s):
  n,m = image.shape
  result = np.empty((n*s,m*s))
  for i in xrange(n):
    for j in xrange(m):
      for k1 in xrange(s):
        for k2 in xrange(s):
          result[i*s+k1,j*s+k2] = image[i,j]
  return result
