import cv2
import cv
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
