import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_brain_contour(image_path, plot=False):
  img = cv2.imread(image_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  gray = cv2.GaussianBlur(gray, (5, 5), 0)
  
  # Brain = White, Background = Black
  thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
  
  thresh = cv2.erode(thresh, None, iterations=2)
  thresh = cv2.dilate(thresh, None, iterations=2)
  
  contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  c = max(contours, key=cv2.contourArea)
  
  extLeft = tuple(c[c[:, :, 0].argmin()][0])
  extRight = tuple(c[c[:, :, 0].argmax()][0])
  extTop = tuple(c[c[:, :, 1].argmin()][0])
  extBot = tuple(c[c[:, :, 1].argmax()][0])
  
  # We add a small pixel buffer so we don't accidentally slice off the brain's edge
  ADD_PIXELS = 0
  new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
  
  final_img = cv2.resize(new_img, (227, 227))
  
  if plot:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original MRI')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped & Resized (227x227)')
    plt.axis('off')
    plt.show()
      
  return final_img

if __name__ == "__main__":
  test_image = "../data/raw/yes/Y1.jpg" 
  processed_image = crop_brain_contour(test_image, plot=True)