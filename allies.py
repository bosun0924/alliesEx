import cv2
import matplotlib.pyplot as plt
import numpy as np

def allies_region(image, mini_map): 
    #get the resolution of the image
    height, width, channel = image.shape
    #set the background of the mask to 0
    mask = np.zeros_like(image)
    #set the cropping polygons
    if (mini_map[0]>0.5*(width)):#if the map is on the right
        ul = (mini_map[0],int(mini_map[1]- (mini_map[2]-mini_map[0])*0.38))
        ur = (1920, int(mini_map[1]- (mini_map[2]-mini_map[0])*0.38))
        dr = (1920, mini_map[1])
        dl = (mini_map[0],mini_map[1])
        area = [ul,ur,dr,dl,]
        crop_area = np.array([area], np.int32)
    if (mini_map[0]<0.5*(width)):
        map_width_limit_left = int((1-map_perc)*width)
        area = [(0, height),(0, map_height_limit),(map_width_limit_left, map_height_limit),(map_width_limit_left, height),]
        crop_area = np.array([area], np.int32)
    
    #get the mask done, the mask only allows minimap area to be further processed
    cv2.fillPoly(mask, crop_area, (255,255,255))
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

'''
class ally:
    def __init__(self, exp):
        self.exp = exp
'''

mini_map = [1655,814,1907,1066]#map location 100%
#mini_map = [1720,880,1910,1069]#map location 0%
imag = cv2.imread('./test3.png')
img = cv2.resize(imag, (1920, 1080))
img_cropped = allies_region(img, mini_map)
gray_img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
himg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img_allies_region = allies_region(img, mini_map)
cir_rad = int((mini_map[3]-mini_map[1])/9)
circles = cv2.HoughCircles(gray_img_cropped,cv2.HOUGH_GRADIENT,1,45,param1=25,param2=17,minRadius=(cir_rad - 2),maxRadius=(cir_rad + 2))
circles = np.uint16(np.around(circles))

#Crop the allies out according to the circle detection results
allies = []
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    # save the area to be an Ally list
    mask = np.zeros_like(himg)
    area = [(i[0],i[1]),(i[0]+i[2],i[1]),(i[0]+i[2],i[1]+i[2]),(i[0],i[1]+i[2]),]
    crop_area = np.array([area], np.int32)
    cv2.fillPoly(mask, crop_area, (255,255,255))
    allies.append(cv2.bitwise_and(himg, mask))

#Extract the number picture out of the ROI(region of interest)


print("_____________________________________")
print(circles)

'''
#############################################################

#h_m = img[62:79, 13:61]
h_m = img[62:79, 78:126]
h_m_hsv = cv2.cvtColor(h_m, cv2.COLOR_BGR2HSV)
#################thresholding to get the bars##################
h_m_health = cv2.inRange(h_m_hsv, (30, 127, 0),(90, 255, 255)) 
h_m_mana = cv2.inRange(h_m_hsv, (90, 127, 0),(150, 255, 255))
#################calculate the percentage####################
h_perc = cv2.mean(h_m_health)[0]
m_perc = cv2.mean(h_m_mana)[0]
print("###############")
print('Health Value', ': ')
print(h_perc)
print('Mana Value', ': ')
print(m_perc)
print("###############")
plt.figure()
plt.imshow(h_m_health)

plt.figure()
plt.imshow(h_m_mana)

plt.figure()
plt.imshow(h_m)

h_m_health = cv2.inRange(himg, (65, 210, 68),(75, 255, 255))
h_m_mana = cv2.inRange(himg, (98, 170, 86),(109, 255, 255))
'''
plt.figure()
plt.imshow(img)

for i in range(4):
    plt.figure()
    plt.imshow(allies[i])
plt.show()

