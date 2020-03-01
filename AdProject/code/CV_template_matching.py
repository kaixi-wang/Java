import cv2
import numpy as np
img = cv2.imread("starbuckCoffee.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("starbuck.png", cv2.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]
result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
# print(result)
loc = np.where(result >= 0.8)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()