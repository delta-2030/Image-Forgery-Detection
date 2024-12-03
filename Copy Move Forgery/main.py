import sys
import cv2
from ForgeryDetection import Detect
import re
from datetime import datetime

file_path = r"E:\DIP Project\Copy Move Forgery\Forged Inputs\test2.png" # Copy the path of the image you want to test
image=cv2.imread(file_path)
eps =60
min_samples=2
print('Detecting Forgery with parameter value as\neps:{}\nmin_samples:{}'.format(eps,min_samples))

detect=Detect(image)

key_points,descriptors = detect.siftDetector() # Coordinates and the 128 dimensional feature vector of the keypoints recognized by SIFT algorithm
sift_image = detect.showSiftFeatures()
forgery=detect.locateForgery(eps,min_samples)
if forgery is None:
	sys.exit(0)
print("Forgery Detected")
cv2.imshow('SIFT Features', sift_image) # Keypoints recognizedd by SIFT
cv2.imshow('Original image',image)
cv2.imshow('Forgery',forgery) # Output - could be mask or forged regions connected by lines.Select definition of locateForgery function accordingly

wait_time=100

while(cv2.getWindowProperty('Forgery', 0) >= 0) and (cv2.getWindowProperty('Original image', 0) >= 0) and (cv2.getWindowProperty('SIFT Features', 0)>=0 ) :
	keyCode = cv2.waitKey(wait_time)
	
cv2.destroyAllWindows()


