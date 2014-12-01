import cv2
import cv
import os
from cesarpy.io import get_all_file_names_from_dir
import numpy as np

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
  def __init__(self, path, ravel=False):
    self.isfile = os.path.isfile(path)
    if self.isfile:
      self.cap = cv2.VideoCapture(path)
      self.nframes = self.cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
    else:
      self.files = get_all_file_names_from_dir(path, sort=True, withpath=True)
      self.nframes = len(self.files)
    self.count = 0 
  def get_next(self):
    if self.isfile:
      ok, frame = self.cap.read()
    else:
      if self.count >= self.nframes:
        ok = False
      else:
        ok = True
        frame = cv2.imread(self.files[self.count])
    if not ok:
      return None
    else:
      self.count += 1
      return frame
