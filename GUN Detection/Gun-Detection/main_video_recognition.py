import cv2
import imutils
import datetime

# Load the trained shape recognition model (replace this with your model loading code)
# model = load_model('path_to_your_trained_model.h5')

gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Access the camera (change the parameter if using a different camera)
camera = cv2.VideoCapture(0)  # Use 0 for the default camera (webcam)

# initialize the first frame in the video stream
firstFrame = None

# flag to track gun detection
gun_exist = False

while True:
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # detect guns in the frame
    gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    if len(gun) > 0:
        gun_exist = True

    # draw rectangles around detected guns
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Perform shape recognition inference using the loaded model
        # Assuming the 'model' object predicts the gun shapes (replace this with your inference code)
        # predicted_shape = model.predict(roi_color)

        # Replace this part with your code to draw the predicted shape on the frame
        # Example code (change this to use the actual predictions from your model)
        predicted_shape = "Weapon"  # Replace this with the actual predicted shape
        cv2.putText(frame, f"Weapon: {predicted_shape}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # draw the text and timestamp on the frame
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # press 'q' to exit the loop
        break

if gun_exist:
    print("guns detected")
else:
    print("guns NOT detected")

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
