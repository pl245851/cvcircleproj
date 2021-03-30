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
print(image.shape)
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
	height, width, channels = image.shape
	pts = np.array(eval("[(10,10),(600,19),(800,800),(100,600)]"), dtype="float32")
	image2 = four_point_transform(image, pts)

	gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
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
		output = four_point_transform(output, pts)
		cv2.circle(output, (maxx, maxy), maxr, (0, 165, 255), 4)
		cv2.rectangle(output, (maxx - 5, maxy - 5), (maxx + 5, maxy + 5), (50, 50, 50), -1)
	# show the output image
	cv2.imwrite("temp.jpeg", output)

	pts1 = "[(0,0),(0,{}),({},{}),({},0)]".format(height, width, height, width)
	pts = np.array(eval(pts1), dtype="float32")
	output = four_point_transform(output, pts)


cv2.imshow("output", np.hstack([image, output]))
cv2.waitKey(0)
