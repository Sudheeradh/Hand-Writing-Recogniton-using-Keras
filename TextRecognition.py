from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import pyfiglet
import time

ascii_banner = pyfiglet.figlet_format("Oculus Alpha")
print(ascii_banner)

def text_detection(scores, box):
	#get the number of rows and columns from scores
	(numRows, numColumns) = scores.shape[2:4]
	rects =[]
	confidences = []

	#looping over rows
	for y in range(0, numRows):
		#extracting bounding boxes and probabilitites
		scoresData = scores[0, 0, y]
		x0 = box[0, 0, y]
		x1 = box[0, 1, y]
		x2 = box[0, 2, y]
		x3 = box[0, 3, y]
		angles = box[0, 4, y]

		#looping over columns
		for x in range(0, numColumns):
			#ignoring scores having insufficient probabilities
			if scoresData[x] < args["min_confidence"]:
				continue

			#resulting feature maps are 4 times smaller, hence we offset
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			#extracting angles
			angle = angles[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			#obtaining height and width of bounding box
			h = x0[x] + x2[x]
			w = x1[x] + x3[x]

			#computing starting and ending co-ordinates of bounding boxes
			endX = int(offsetX + (cos*x1[x]) + (sin*x2[x]))
			endY = int(offsetY - (sin*x1[x]) + (cos*x2[x]))

			startX = int(endX - w)
			startY = int(endY - h)


			#adding bounding box co-ordinates and scores to lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	return (rects, confidences)


	#using argument parser

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, help = "path to input image")
ap.add_argument("-east", "--east", type = str, help = "path to East model")
ap.add_argument("-c", "--min_confidence", type = float, default = 0.9, help = "minimum probability required to inspect a region")

#The image dimensions should be in multiples of 32 because the EAST model requires it

ap.add_argument("-w", "--width", type = int, default = 320, help = "resized image width(multiple of 32)")
ap.add_argument("-e", "--height", type = int, default = 320, help = "resized image height(multiple of 32)")
ap.add_argument("-p", "--padding", type = float, default = 0.0, help = "amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


# loading the image
image = cv2.imread(args["image"])
orig = image.copy()

#obtaining the dimensions
(origH, origW) = image.shape[:2]

#setting new width and obtaining ratio

(newWidth, newHeight) = (args["width"], args["height"])
rW = origW / float(newWidth)
rH = origH / float(newHeight)


#Resizing the image
image = cv2.resize(image, (newWidth, newHeight))
(H, W) = image.shape[:2]


#Getting the output from the EAST model
#Probabilities and bounding boxes

layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

#Loading the pre-trained EAST model

print("loading EAST detector :)")
net = cv2.dnn.readNet(args["east"])

#constructing a blob from the image

blob = cv2.dnn.blobFromImage(image, 1.0, (W,H), (123.68, 116.78, 103.94), swapRB = True, crop = False)


#performing forward pass through Deep learning model and obtaining the outputs

start = time.time()
net.setInput(blob)
(scores, box) = net.forward(layerNames)
end = time.time()

print("Time taken for detection is " + str(end - start) + "   :)")


#text detection
(rects, confidences) = text_detection(scores, box)
boxes = non_max_suppression(np.array(rects), probs = confidences)


#initialize the list of results
results = []

#looping over bounding boxes
for (startX, startY, endX, endY) in boxes:
	'''
	#scaling bounding boxes based on ratios
	startX = int(startX * ratioWidth)
	startY = int(startY * ratioHeight)
	endX = int(endX * ratioWidth)
	endY = int(endY * ratioHeight)
	
	#obtaining padding
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	#applying padding
	startX = max(0, startX - dX)
	startY = max(0, startX - dY)
	endX = min(originalWidth, endX + (dX * 2))
	endY = min(originalHeight, endY + (dY * 2))
	
	#Extracting the padded Regio Of Interest(ROI)
	roi = orig[startY:endY, startX:endX]
	'''
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	# in order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	# apply padding to each side of the bounding box, respectively
	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	# extract the actual padded ROI
	roi = orig[startY:endY, startX:endX]

	#Configuring tesseract
	config = ("-l eng --oem 1 --psm 11")
	OCR = pytesseract.image_to_string(roi, config = config)

	#adding bounding boxes and OCR text into list of results
	results.append(((startX, startY, endX, endY), OCR))



#Displaying the results

#sorting the results from top to bottom
results = sorted(results, key = lambda r:r[0][1])

#looping over the results
for ((startX, startY, endX, endY), OCR) in results:
	#displaying the text detected by Tesseract
	print("Detected Text")
	print("^_^  ================  ^_^")
	print("{}\n".format(OCR))
	print("^_^  ================  ^_^")

	#Removing non standard text and outputting text and bounding boxes
	text = "".join([c if ord(c) < 128 else " " for c in OCR]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.putText(output, OCR, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

	#Show output imagw
	cv2.imshow("Oculus Alpha", output)
	cv2.waitKey(0)



