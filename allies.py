import cv2
import matplotlib.pyplot as plt
import numpy as np

def allies_region(image, mini_map): 
    #get the resolution of the image
    height, width = image.shape
    #set the cropping polygons
    if (mini_map[0]>0.5*(width)):#if the map is on the right
        map_width_limit = int(map_perc*width)
        area = [(map_width_limit, height),(map_width_limit, map_height_limit),(width, map_height_limit),(width, height),]
        crop_area = np.array([area], np.int32)
    if corner == 'left':
        map_width_limit_left = int((1-map_perc)*width)
        area = [(0, height),(0, map_height_limit),(map_width_limit_left, map_height_limit),(map_width_limit_left, height),]
        crop_area = np.array([area], np.int32)
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
'''
class ally:
    def __init__(self, id_cir, ):
'''



mini_map = [1655,814,1907,1066]#map location
img = cv2.imread('./test.png')
img = cv2.resize(img, (1920, 1080))
img = img[int(mini_map[1]*0.88):int(mini_map[1]*0.983), mini_map[0]:1920]
'''
cir_rad = int((mini_map[3]-mini_map[1])/9)
cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_allies_region = allies_region(img, mini_map)
circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,param1=30,param2=20,minRadius=(cir_rad - 2),maxRadius=(cir_rad + 2))

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(255,255,255),1)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

print("_____________________________________")
print(circles)
'''
h_m = img[62:79, 13:61]
h_m_hsv = cv2.cvtColor(h_m, cv2.COLOR_BGR2HSV)
h_m_health = cv2.inRange(h_m_hsv, (30, 127, 0),(90, 255, 255)) 
h_m_mana = cv2.inRange(h_m_hsv, (90, 127, 0),(150, 255, 255))

plt.figure()
plt.imshow(h_m_health)

plt.figure()
plt.imshow(h_m_mana)

plt.figure()
plt.imshow(h_m)

plt.show()
