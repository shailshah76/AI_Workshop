# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:31:25 2018

@author: Dell
"""
import cv2

image=cv2.imread("standard_test_images/Lena_color_512.tif")#to read a image
image=image*1
cv2.imshow("LENA",image)#to show the image
cv2.waitKey()#to hold the image for specific time
cv2.destroyAllWindows()#to destroy all the windows
#print(image)

rgb_img=cv2.imread("standard_test_images/Lena_color_512.tif")
gray=cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
#to convert the image from bgr to gray
hsv_img=cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)#to covert  from bgr to hsv
cv2.imshow("Lena", gray)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow("Lena", hsv_img)
cv2.waitKey()
cv2.destroyAllWindows()

rect_img=cv2.rectangle(rgb_img,(70,70),(400,400),(215,0,123),10)
#to enter shape on image
cv2.imshow("Lena", rect_img)
cv2.waitKey()
cv2.destroyAllWindows()

text_img=cv2.putText(rgb_img,"LENA", (100,100),cv2.FONT_HERSHEY_COMPLEX,
                     2.5,(0,255,0),3)#to add text on the image
cv2.imshow("Lena", text_img)
cv2.waitKey()
cv2.destroyAllWindows()

bk_pg=cv2.imread("standard_test_images/bookpage.jpg",0)
ret_value,thres=cv2.threshold(bk_pg,9,150,cv2.THRESH_BINARY)
#removes unwanted noise from pcture
cv2.imshow("Thresholded Image", thres)
cv2.waitKey()
cv2.destroyAllWindows()

adaptive_thresh=cv2.adaptiveThreshold(bk_pg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                                       , cv2.THRESH_BINARY,115,1)
#120 is filter size
#guassian is the method for thresholding n binary is the output form
cv2.imshow(" ADAPTIVE Thresholded Image", adaptive_thresh)
cv2.waitKey()
cv2.destroyAllWindows()