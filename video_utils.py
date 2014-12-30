import cv2
import cv
import os
from cesarpy.io import get_all_file_names_from_dir
import numpy as np
import utils

def get_right_empty(h, w, nframes, want_gray, ravel, dtype):
  if want_gray:
    if ravel:
      frames = np.empty(( nframes, h*w), dtype=dtype)
    else:
      frames = np.empty(( nframes, h, w), dtype=dtype)
  else:
    if ravel:
      frames = np.empty(( nframes, h*w*3), dtype=dtype)
    else:
      frames = np.empty(( nframes, h, w, 3), dtype=dtype)
  return frames


def get_all_frames(path, want_gray=True, ravel=True, dtype=np.uint8):
  cap = cv2.VideoCapture(path)
  nframes = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
  w = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
  frames = None
  count = 0 
  while(True):
    ret, frame = cap.read()
    if not ret:
      break
    if frames == None:
      already_gray = len(frame.shape) == 2
      if not want_gray and already_gray:
        want_gray = True
      frames = get_right_empty(h, w, nframes, want_gray, ravel, dtype)
    if want_gray:
      gray_frame = frame if already_gray else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      if ravel:
        frames[count,:] = gray_frame.ravel()
      else:
        frames[count,:,:] = gray_frame
    else:
      if ravel:
        frames[count,:] = frame.ravel()
      else:
        frames[count,:,:,:] = frame
    count += 1
  # When everything done, release the capture
  cap.release()
  return frames

class VideoIter:
  def __init__(self, path, ravel=False, use_skimage=False, gray=False, inds=None, starting_from=0):
    self.inds = inds
    self.gray = gray
    self.use_skimage = use_skimage
    self.isfile = os.path.isfile(path)
    if self.isfile:
      if use_skimage:
        raise Exception('If you are reading from a video file you cannot use skimage.')
      self.cap = cv2.VideoCapture(path)
      self.nframes = int(self.cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
      self.count = 0
      if self.inds == None and starting_from != 0:
        while self.count < starting_from:
          ok, frame = self.cap.read()
          self.count += 1
    else:
      self.files = get_all_file_names_from_dir(path, sort=True, withpath=True)
      self.nframes = len(self.files)
      self.count = starting_from
    self.meta_idx = 0
  def __iter__(self):
    return self
  def next(self):
    if self.isfile:
      ok, frame = self.cap.read()
      if self.inds != None:
        if self.meta_idx < len(self.inds):
          while self.count < self.inds[self.meta_idx]:
            self.count += 1
            ok, frame = self.cap.read()
        else:
          ok = False
      if ok and self.gray:
        frame = utils.bgr2gray(frame)
    else:
      if self.count >= self.nframes or (self.inds != None and self.meta_idx >= len(self.inds)):
        ok = False
      else:
        ok = True
        idx = self.inds[self.meta_idx] if self.inds != None else self.count
        frame = utils.load_img(self.files[idx], gray=self.gray, use_skimage=self.use_skimage)
    if not ok:
      raise StopIteration
    else:
      self.meta_idx += 1
      self.count += 1
      return frame
