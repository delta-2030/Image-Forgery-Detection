import numpy as np
import cv2
from sklearn.cluster import DBSCAN

class Detect(object):
    def __init__(self, image):
        self.image = image

    def siftDetector(self):
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.key_points, self.descriptors = sift.detectAndCompute(gray, None)
        return self.key_points, self.descriptors

    def showSiftFeatures(self):
        sift_image = cv2.drawKeypoints(self.image, self.key_points, self.image.copy())
        return sift_image

    def locateForgery(self, eps, min_sample):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(self.descriptors)
        size = np.unique(clusters.labels_).shape[0] - 1
        forgery = self.image.copy()

        if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
            print('No Forgery Found!!!')
            return None

        if size == 0:
            size = 1

        cluster_list = [[] for _ in range(size)]
        for idx in range(len(self.key_points)):
            if clusters.labels_[idx] != -1:
                cluster_list[clusters.labels_[idx]].append(
                    (int(self.key_points[idx].pt[0]), int(self.key_points[idx].pt[1]))
                )

        for points in cluster_list:
            if len(points) > 1:
                for idx1 in range(1, len(points)):
                    cv2.line(forgery, points[0], points[idx1], (255, 0, 0), 5)

        return forgery

    def locateForgery(self, eps, min_sample):
        clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(self.descriptors)
        size = np.unique(clusters.labels_).shape[0] - 1 
        if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
            print('No Forgery Found!!!')
            return None
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cluster_list = [[] for _ in range(size)]
        for idx in range(len(self.key_points)):
            if clusters.labels_[idx] != -1:
                cluster_list[clusters.labels_[idx]].append(
                    (int(self.key_points[idx].pt[0]), int(self.key_points[idx].pt[1]))
                )
        for points in cluster_list:
            if len(points) > 1:
                for x, y in points:
                    cv2.circle(mask, (x, y), radius=1, color=255, thickness=-1)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask = cv2.dilate(mask, kernel_dilate, iterations=2) 
        mask = cv2.erode(mask, kernel_erode, iterations=1) 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)

        return refined_mask
