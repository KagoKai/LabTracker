import cv2

img = cv2.imread('cat.jpg')
(h, w, _) = img.shape

cv2.imshow('test', img)
cv2.waitKey(0)

cv2.rectangle(img, (-9, 200), (109, 300), color=(0,0,255), thickness=2)
cv2.imshow('test', img)
cv2.waitKey(0)

cat = img[-9:109, 200:300]
cv2.imshow('cat', cat)
cv2.waitKey(0)