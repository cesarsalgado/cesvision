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

def put_in_255_range(im):
  maxv = im.max()
  minv = im.min()
  result = 255*((im-minv)/float(maxv-minv))
  result = result.astype(np.uint8)
  return result 

def rectify(img):
  n,m = img.shape
  for i in xrange(n):
    for j in xrange(m):
      v = img[i,j]
      if v > 255:
        img[i,j] = 255
      elif v < 0:
        img[i,j] = 0
  return img

def psnr(original, other):
  original = original.astype(float)/255
  other = other.astype(float)/255
  return -10*np.log10(np.mean((original-other)**2))

def add_noise(img, sigma):
  return rectify(img.astype(float)+sigma*np.random.randn(*img.shape)).astype(np.uint8)

def show_image2(img):
  if len(img.shape) == 2:
    plt.imshow(img, cmap=plt.cm.gray, interpolation='none')
  else:
    plt.imshow(img, interpolation='none')
  plt.show()

def dall():
  cv2.destroyAllWindows()

def to_gray(im):
  if im.dtype != np.uint8:
    imdtype = im.dtype
    im = im.astype(np.uint8)
  else:
    imdtype = np.uint8
  result = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
  if imdtype != np.uint8:
    result = result.astype(imdtype)
  return result

def load_img(path, gray=False, to_float=False):
  if gray:
    img = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  else:
    img = cv2.imread(path)
  if to_float:
    img = img.astype(float)
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
  if s == 1:
    return image.copy()
  n,m = image.shape[:2]
  if len(image.shape) == 3:
    result = np.empty((n*s,m*s,image.shape[2]))
  else:
    result = np.empty((n*s,m*s))
  for i in xrange(n):
    for j in xrange(m):
      i_s, j_s = i*s, j*s
      result[i_s:i_s+s,j_s:j_s+s] = image[i,j]
  return result
