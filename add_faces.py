# Import the OpenCV library
import cv2

# Open the video capture device (webcam)
video = cv2.VideoCapture(0)

# Load the pre-trained Haar cascade classifier for face detection
face_detect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# Loop to continuously capture frames from the video feed
while True:
    # Read a frame from the video feed
    ret, frame = video.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    
    # Display the frame with face detection
    cv2.imshow("frame", frame)
    
    # Check for key press events
    k = cv2.waitKey(1)
    
    # If the user presses 'q' or 'Q', break out of the loop
    if k == ord("Q") or k == ord("q"):
        break

# Release the video capture device
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()