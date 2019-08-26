import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract

#mini_map = [1655,814,1907,1066]#map location 100%
mini_map = [1720,880,1910,1069]#map location 0%
imag = cv2.imread('./test4.png')
img = cv2.resize(imag, (1920, 1080))
#cropping the allies area
if (mini_map[0]>960):
	img_cropped = img[int(mini_map[1]- (mini_map[2]-mini_map[0])*0.38):mini_map[1],mini_map[0]:1920]
else:
	img_cropped = img[int(mini_map[1]- (mini_map[0]-mini_map[2])*0.38):mini_map[1],0:mini_map[0]]

gray_img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
himg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
himg_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
#img_allies_region = allies_region(img, mini_map)
cir_rad = int((mini_map[3]-mini_map[1])/9)
circles = cv2.HoughCircles(gray_img_cropped,cv2.HOUGH_GRADIENT,1,45,param1=23,param2=15,minRadius=(cir_rad - 2),maxRadius=(cir_rad + 2))
circles = np.uint16(np.around(circles))
print("_____________________________________")
print(circles)
#########trial###########
'''
font_low = (0,0,170)
font_high = (180,30,255)
numbers = cv2.inRange(himg_cropped, font_low, font_high)

cv2.imwrite('train.png',numbers)
print(pytesseract.image_to_string(numbers))
'''

#Crop the allies out according to the circle detection results
allies = []
allies_c = []
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img_cropped,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_cropped,(i[0],i[1]),2,(0,0,255),3)
    # crop the area
    area = himg_cropped[i[1]:i[1]+i[2], i[0]:i[0]+i[2]]
    # save the area to be an Ally list
    allies.append([i[0],i[1],i[2],area])
    allies_c.append([i[0],i[1],i[2],area])

#Extract the number picture out of the ROI(region of interest)
#In the HSV space, the exp font is in range (0,0,118) to (180, 28, 255)
font_low = (0,0,165)
font_high = (180,15,255)

for i in range(4):
    allies_c[i][3] = cv2.inRange(allies_c[i][3], font_low, font_high)
    height, width = allies_c[i][3].shape
    allies_c[i][3] = cv2.resize(allies_c[i][3],(height*4, width*4))
    cv2.imwrite('train.png',allies_c[i][3])
    print(pytesseract.image_to_string(allies_c[i][3]))
#'''
'''
#############################################################
'''
plt.figure()
plt.imshow(himg_cropped)

plt.figure()
plt.imshow(img_cropped)

for i in range(4):
    plt.figure()
    plt.imshow(allies[i][3])

for i in range(4):
    plt.figure()
    plt.imshow(allies_c[i][3])

plt.show()

