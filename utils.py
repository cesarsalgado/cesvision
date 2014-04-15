import cv2

def show_image(img, name='image'):
  cv2.imshow(name, img)
  cv2.waitKey(-1)

def dall():
  cv2.destroyAllWindows()
