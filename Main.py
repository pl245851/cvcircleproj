import cv2
import numpy as np
im = input("image file name\n")
image = cv2.imread(im, cv2.IMREAD_COLOR)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 100)
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	maxx = 0
	maxy = 0
	maxr = 0
	for (x, y, r) in circles:
		print(x,y,r)
		if r > maxr:
			maxx = x
			maxy = y
			maxr = r
	# draw the circle in the output image, then draw a rectangle
	# corresponding to the center of the circle
	cv2.circle(output, (maxx, maxy), maxr, (0, 165, 255), 4)
	cv2.rectangle(output, (maxx - 5, maxy - 5), (maxx + 5, maxy + 5), (50, 50, 50), -1)
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)
