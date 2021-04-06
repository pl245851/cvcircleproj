import cv2
import numpy as np


# from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
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
#pertran = input("change perspective\n")
image = cv2.imread(im, cv2.IMREAD_COLOR)
output = image.copy()
height, width, channels = image.shape
print(image.shape)
if(height != 533 and width != 800):
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
	#for x in range(200, height*2,10):
	#	for y in range(1150, 1160,10):
	#		print(x,y)
	#while True:
		#x = input("x")#270
		#y = input("y")#1320
			cv2.destroyAllWindows()
			ptsstring = "[(0,0),(0,{}),({},{}),({},0)]".format(733,width,height,700)
			pts = np.array(eval(ptsstring), dtype="float32")
			pts1 = "[(0,0),(0,{}),({},{}),({},0)]".format(533, 800, 533, 800)
			pts2 = np.array(eval(pts1), dtype="float32")
			image2 = four_point_transform(image, pts)
			#image2 = four_point_transform(image2, pts2)
#			image2 = four_point_transform(image2, pts)

			#cv2.imshow("output", image2)
				#	print(x,y)
				#	cv2.waitKey(10)
			#cv2.waitKey(0)

			gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(gray, (5, 5), 0)
			#cv2.imshow("output", blur)
			circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 100)
			# ensure at least some circles were found
			blank_image = np.zeros((height, width, 3), np.uint8)
			blank_image = four_point_transform(blank_image, pts)
			if circles is not None:
				# convert the (x, y) coordinates and radius of the circles to integers
				circles = np.round(circles[0, :]).astype("int")
				# loop over the (x, y) coordinates and radius of the circles
				maxx = 0
				maxy = 0
				maxr = 0
				for (x2, y2, r) in circles:
					#print(x, y, r)
					if r > maxr:
						maxx = x2
						maxy = y2
						maxr = r
				# draw the circle in the output image, then draw a rectangle
				# corresponding to the center of the circle
				blank_image = four_point_transform(blank_image, pts)

				#cv2.imshow("blank", blank_image)
				cv2.circle(blank_image, (maxx, maxy), maxr, (0, 165, 255), 4)
				cv2.rectangle(blank_image, (maxx - 5, maxy - 5), (maxx + 5, maxy + 5), (50, 50, 50), -1)
				#cv2.imshow("circle", blur)
				#cv2.imwrite("{}_{}.jpeg".format(x,y), blur)
			#cv2.imshow("output", np.hstack([blank_image, output]))
				#cv2.waitKey(0)
	# show the output image
	#cv2.imwrite("temp.jpeg", output)

#			pts1 = "[(0,0),(0,{}),({},{}),({},0)]".format(533, 800, 533, 800)
#			pts2 = np.array(eval(pts1), dtype="float32")
			#cv2.imshow(" ", blank_image)
			#print(pts)
			#inv_trans = np.array(np.linalg.inv(pts), dtype="float32")
			#print(inv_trans)
			blank_image = four_point_transform(blank_image, pts2)
			print(blank_image.shape)

			cv2.waitKey(0)
			#cv2.imshow("circletilt", blank_image)
			output = cv2.add(blank_image, output)

#cv2.imshow("image2", image2)
#cv2.imshow("circle", blank_image)
cv2.imshow("output", np.hstack([image, output]))
cv2.waitKey(0)