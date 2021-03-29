import cv2
import numpy as np

# from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

# from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


im = input("image file name\n")
pertran = input("change perspective\n")
image = cv2.imread(im, cv2.IMREAD_COLOR)
output = image.copy()
if(pertran != "yes"):
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
			print(x, y, r)
			if r > maxr:
				maxx = x
				maxy = y
				maxr = r
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (maxx, maxy), maxr, (0, 165, 255), 4)
		cv2.rectangle(output, (maxx - 5, maxy - 5), (maxx + 5, maxy + 5), (50, 50, 50), -1)
	# show the output image
else:

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = 255 - gray

	# blur image
	blur = cv2.GaussianBlur(gray, (3, 3), 0)

	# do adaptive threshold on gray image
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 2)
	thresh = 255 - thresh

	# apply morphology
	kernel = np.ones((5, 5), np.uint8)
	rect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel)

	# thin
	kernel = np.ones((5, 5), np.uint8)
	rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

	# get largest contour
	contours = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	for c in contours:
		area_thresh = 0
		area = cv2.contourArea(c)
		if area > area_thresh:
			area = area_thresh
			big_contour = c

	# get rotated rectangle from contour
	rot_rect = cv2.minAreaRect(big_contour)
	box = cv2.boxPoints(rot_rect)
	box = np.int0(box)
	for p in box:
		pt = (p[0], p[1])
		print(pt)

	# draw rotated rectangle on copy of img
	rot_bbox = image.copy()
	cv2.drawContours(rot_bbox, [box], 0, (0, 0, 255), 2)

	# write img with red rotated bounding box to disk
	#cv2.imwrite("rectangle_thresh.png", thresh)
	#cv2.imwrite("rectangle_outline.png", rect)
	#cv2.imwrite("rectangle_bounds.png", rot_bbox)

	# display it
	#cv2.imshow("IMAGE", image)
	#cv2.imshow("THRESHOLD", thresh)
	cv2.imshow("RECT", rect)
	cv2.imshow("BBOX", rot_bbox)
	cv2.waitKey(0)

#	cv2.imshow("output", np.hstack([image, output]))
#	cv2.waitKey(0)
