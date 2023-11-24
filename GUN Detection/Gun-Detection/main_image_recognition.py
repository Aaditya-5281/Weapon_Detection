import numpy as np
import cv2
import imutils
import datetime

gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Read the image file
image = cv2.imread(r"C:\Users\aadit\Downloads\pexels-specna-arms-889709.jpg")  # Replace 'path_to_your_image.jpg' with your image file path

# Convert the image to grayscale and blur it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)

# Detect guns in the image
guns = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

# Draw rectangles around detected guns
for (x, y, w, h) in guns:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Add timestamp to the image
timestamp = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
cv2.putText(image, timestamp, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

# Resize the image to a smaller size for display
resized_image = imutils.resize(image, width=800)  # Adjust the width as needed to make the image smaller

# Display the processed image
cv2.imshow("Processed Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
