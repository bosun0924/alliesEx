import cv2
import pytesseract
import numpy as np
from PIL import Image

img = cv2.imread('./train.png')
height, width, _ = img.shape
#increase resolution
img = cv2.resize(img, (width*4, height*4))
#blur the edges
img = cv2.blur(img, (5,5))
#get a smoothed font
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#thining the edges
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(thresh1,kernel,iterations = 1)
#try OCR
print(pytesseract.image_to_string(img))
cv2.imwrite('./train0.png', thresh1)
cv2.imshow('original', img)
cv2.imshow('result', thresh1)
cv2.imshow('erosion', erosion)
cv2.waitKey(0)
#print(pytesseract.image_to_string(img))