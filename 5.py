from typing import final
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('./Picture1.jpg')
height, width = img.shape[:2]

mask = np.zeros(img.shape[:2],np.uint8)
bgmodel = np.zeros((1,65),np.float64)

fgmodel = np.zeros((1,65),np.float64)

rect = (15, 15, width-30, height-10)

cv.grabCut(img, mask, rect, bgmodel, fgmodel, 5, cv.GC_INIT_WITH_RECT)

mask = np.where((mask==2) | (mask==0),0,1).astype('uint8')

img_1 = img*mask[:,:,np.newaxis]

bg = img - img_1
img_2 = img - bg

img_2[np.where((img_2 > [0,0,0]).all(axis = 2))] = [255,255,255]
cpl2 = bg + img_2

bg[np.where((bg > [0,0,0]).all(axis = 2))] = [255,255,255]
cpl = bg + img_1


#plt.hist(img1.ravel(),256,[0,256]); plt.show()

cv.imshow('Output1', cpl)
cv.imshow('Output2', cpl2)
key = cv.waitKey(0)
if key == 2:
 cv.destroyAllWindows()
