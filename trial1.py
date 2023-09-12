#!/usr/bin/env python3
import cv2 as cv
import numpy as np

def sort(arr_i,arr_j,choice):
     pass

image = cv.imread("test/2.png")
cv.imshow('image', image)
h,w,ch = image.shape
print(h,w)
arr_i = []
arr_j = []
# print(image[1,3])
for i in range(h):
        for j in range(w):
            if image[i,j][0] == 255 and image[i,j][1] < 255 and image[i,j][2] < 255:
                # image[i,j] = [255,0,0]
                arr_i.append(i)
                arr_j.append(j)
            # else :
                # image[i,j] = [0,0,0]
# print(arr_i,arr_j)
min_ver = [arr_i[0], arr_j[0]]
max_ver = [arr_i[len(arr_i) - 1],arr_j[len(arr_i) - 1] ]
min_hor = [ arr_i[arr_j.index(min(arr_j))], min(arr_j)]
max_hor = [ arr_i[arr_j.index(max(arr_j))], max(arr_j)]
print(min_hor, max_hor, min_ver, max_ver)
min_hor.reverse()
max_hor.reverse()
min_ver.reverse()
max_ver.reverse()
# cv.circle(image, min_hor, 2, (0,0,255), thickness=-1)
# cv.circle(image, max_hor, 2, (0,0,255), thickness=-1)
# cv.circle(image, min_ver, 2, (0,0,255), thickness=-1)
# cv.circle(image, max_ver, 2, (0,0,255), thickness=-1)
pts_ground = np.float32([min_ver,max_ver,min_hor,max_hor])
pts_trans = np.float32([[300,200],[300,400],[200,300],[400,300]])
mat = cv.getPerspectiveTransform(pts_ground,pts_trans)
res = cv.warpPerspective(image, mat, (600,600))
# cv.imshow('blue', image)

cv.imshow('mod',res)
cv.waitKey(0)
cv.DestroyAllWindows()



image = cv.imread("test/2.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 1,255,cv.THRESH_BINARY)
# thresh = np.uint8(border_maker(thresh))
print(thresh)
thresh = cv.Canny(thresh, 50,150,apertureSize = 3)
# print(thresh)
# cv.imshow('corner', thresh)

corners = np.intp(cv.goodFeaturesToTrack(thresh, 4, 0.01, 10))
print(corners)
arr = []
for i in corners:
    x,y = i.ravel()
    arr.append([x-2,y-2])
    cv.circle(image, (x-2,y-2),2, (0,0,255), -1)

print(arr)
arr2 = quadrant_finder(arr,image.shape[0], image.shape[1])
print(arr2)

matrix = cv.getPerspectiveTransform(np.float32(arr),np.float32(arr2))
frame = cv.warpPerspective(image, matrix,(600,600))
cv.imshow('final', frame)


# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# result = cv2.warpPerspective(frame, matrix, (500, 600))

    # print(x-2,y-2)
    # cv.circle(image, (x-2,y-2),2, (0,0,255), -1)

# contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(image, contours, -1, (255,0,0), 2)
# print(contours)
cv.imshow('binary', image)
