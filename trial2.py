#!/usr/bin/env python3
import cv2 as cv
import numpy as np

def border_maker(image):
    arr = np.asarray(image)
    arr2 = np.zeros((arr.shape[0] + 4, arr.shape[1] + 4))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr2[i+2,j+2] = arr[i,j]
    return arr2.astype('int32')

def quadrant_finder(corners,h,w):
    arr = []
    for i in corners:
        # print(i, i[0])
        if i[0] < h/2 and i[1] < w/2:
            arr.append([0,0])
        elif i[0]<h/2 and i[1] > w/2:
            arr.append([0,599])
        elif i[0]>h/2 and i[1] < w/2:
            arr.append([599,0])
        else:
            arr.append([599,599])
    return arr

image= cv.imread("test/2.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #grayscaling
ret, thresh = cv.threshold(gray, 1,255,cv.THRESH_BINARY) #binary thresholding 
thresh = np.uint8(border_maker(thresh)) #frame expansion
canny = cv.Canny(thresh, 50,150,apertureSize = 3) #canny edge detection (forms outer frame)
corners = np.intp(cv.goodFeaturesToTrack(canny, 4, 0.01, 10)) #corner detection

src = []
for i in corners:  # corner data extraction for perspective transform
    x,y = i.ravel()
    src.append([x-2,y-2])
dst = quadrant_finder(src,image.shape[0], image.shape[1])

matrix = cv.getPerspectiveTransform(np.float32(src),np.float32(dst))
frame = cv.warpPerspective(image, matrix,(600,600))
cv.imshow('image', frame)

if cv.waitKey(0) & 0xff == 27:
    cv.DestroyAllWindows()
