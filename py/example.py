import cv2
import numpy as np

class Bbox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def __str__(self):
        return f'x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}'


class PanelDetectorHsv:
    def __init__(self, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max

    def hsv_filter(self, img: np.ndarray):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_min, self.sat_min, self.val_min])
        upper = np.array([self.hue_max, self.sat_max, self.val_max])
        mask = cv2.inRange(img_hsv, lower, upper)
        return cv2.bitwise_and(img, img, mask=mask)

    def detect(self, img: np.ndarray) -> list:
        img_filtered = self.hsv_filter(img)
        img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = self.create_bboxes(contours)
        merge_contours = []
        while len(bbox_list) > 0:
            bbox_list, box = self.find_nearby_contours(bbox_list, 10, 10)
            merge_contours.append(box)
        return merge_contours

    def find_nearby_contours(self, bbox_list: list, x_mergin: int, y_mergin: int):
        box = Bbox(0, 0, 0, 0)

        if len(bbox_list) == 1:
            return bbox_list, bbox_list[0]

        for i in range(len(bbox_list)):
            for j in range(i + 1, len(bbox_list)):
                if abs(bbox_list[i].x - bbox_list[j].x) < x_mergin and abs(bbox_list[i].w - bbox_list[j].w) < y_mergin:
                    box.x = min(bbox_list[i].x, bbox_list[j].x)
                    box.y = min(bbox_list[i].y, bbox_list[j].y)
                    box.w = max(bbox_list[i].w, bbox_list[j].w)
                    box.h = abs(bbox_list[i].y - bbox_list[j].y) + max(bbox_list[i].h, bbox_list[j].h)
                    bbox_list.remove(bbox_list[i])
                    bbox_list.remove(bbox_list[j - 1])
                    return bbox_list, box

            bbox_list.remove(bbox_list[i - 1])
            if box.w != 0:
                break

        return bbox_list, box
    
    def draw_bbox(self, img: np.ndarray, bbox_list: list):
        for box in bbox_list:
            cv2.rectangle(img, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 255, 0), 2)
        return img

    def create_bboxes(self, contours):
        bbox_list = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox_list.append(Bbox(x, y, w, h))
        return bbox_list

img_path = 'image_0316.png'
img = cv2.imread(img_path)

panel_detector = PanelDetectorHsv(80, 150, 50, 255, 50, 255)
bboxes = panel_detector.detect(img)
img = panel_detector.draw_bbox(img, bboxes)

cv2.imshow('img', img)
cv2.waitKey(0)
