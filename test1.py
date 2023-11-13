#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import math as m

def gauss(spatial_gauss_coeff, intensity_gauss_coeff,k):
    arr_spatial= np.zeros((k,k))
    sp_sq = m.pow(spatial_gauss_coeff,2)
    arr_intensity = np.zeros(256)
    in_sq = m.pow(intensity_gauss_coeff,2)
    i, j = np.indices((k, k))
    arr_spatial = (1 / (2 * m.pi * sp_sq)) * np.exp(-((i - (k - 1) / 2)**2 + (j - (k - 1) / 2)**2) / (2 * sp_sq))
    indices = np.arange(256)
    arr_intensity = (1 / (2 * m.pi * in_sq)) * np.exp(-(indices**2) / (2 * in_sq))
    return arr_spatial,arr_intensity

def bilateral_mask_with_cross(p,q,M,arr_spatial, arr_intensity, ambient,flash,k,n):
    # global arr_spatial
    # global arr_intensity
    # global ambient
    # global flash
    # global k
    # global n
    rgb_bilateral = np.zeros(3)
    rgb_cross = np.zeros(3)
    for r in range(0,3,1):     #individual for BGR
        arr = np.zeros((k,k,4))
        for i in range(0,k,1):
            for j in range(0,k,1):
                Is = ambient[(p-n+i),(q-n+j), r]
                Is_flash = flash[(p-n+i),(q-n+j), r]
                Ip = ambient[p,q,r]
                Ip_flash = flash[p,q,r]
                arr[i,j,0] = Is*arr_spatial[i,j]*arr_intensity[abs(Is - Ip)]
                arr[i,j,1] = arr_spatial[i,j]*arr_intensity[abs(Is - Ip)]
                arr[i,j,2] = Is*arr_spatial[i,j]*arr_intensity[abs(Is_flash - Ip_flash)]
                arr[i,j,3] = arr_spatial[i,j]*arr_intensity[abs(Is_flash - Ip_flash)]
        rgb_bilateral[r] = int(m.floor(np.sum(arr[:,:,0]) / np.sum(arr[:,:,1])))
        rgb_cross[r] = int(m.floor(np.sum(arr[:,:,2]))/np.sum(arr[:,:,3]))
    return (M*rgb_bilateral + (1-M)*rgb_cross).astype(int)


ambient = cv.imread('/home/anunay/assgn2_ee604/Q2/ultimate_test/2_a.jpg')
flash = cv.imread('/home/anunay/assgn2_ee604/Q2/ultimate_test/2_b.jpg')
# cv.imshow('spatial',ambient)
spatial_gauss_coeff = 2
n = int(m.floor(m.pi*spatial_gauss_coeff))
k = 2*n + 1
intensity_gauss_coeff = 10
arr_spatial,arr_intensity = gauss(spatial_gauss_coeff,intensity_gauss_coeff,k)
print(arr_intensity,arr_spatial)
# cv.imshow('gauss', arr_spatial*255)
# arr = np.zeros((ambient.shape[0] - (k-1), ambient.shape[1] - (k-1), 3))
# for i in range(n,(ambient.shape[0] - n),1):
#     for j in range(n,(ambient.shape[1] - n),1):
#         arr[i-n,j-n,:] = bilateral_mask_with_cross(i,j,0.5,arr_spatial,arr_intensity,ambient,flash,k,n)
#         print(arr[i-n,j-n,:])
# print(arr)
# cv.imshow('bilateral', arr)
# if cv.waitKey(0) & 0xff == 27:
#     cv.DestroyAllWindows()