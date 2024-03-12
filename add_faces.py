import cv2
import pickle
import numpy as np
import os

# Open the webcam
cam = cv2.VideoCapture(0)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Initialize a list to store face images
faces = []

# Counter for the number of captured face images
counter = 0

# Get the child's name
child_name = input("Enter your name: ")

# Loop to capture face images
while True:
    ret, frame = cam.read()  # Read a frame from the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale frame
    
    # Loop through each detected face
    for (x, y, w, h) in detected_faces:
        # Crop the detected face region
        face_img = frame[y:y+h, x:x+w, :]
        # Resize the cropped face image to a standard size
        resized_face = cv2.resize(face_img, (50, 50))
        # Add the resized face image to the list
        if len(faces) < 100 and counter % 10 == 0:
            faces.append(resized_face)
        counter += 1
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        # Display the count of captured face images
        cv2.putText(frame, str(len(faces)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 1)
    
    # Display the frame with face detection
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)  # Check for key press events
    
    # If the user presses 'q' or the maximum number of face images is captured, break out of the loop
    if key == ord('q') or key == ord('Q') or len(faces) == 100:
        break

# Release the webcam
cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Convert the list of face images to a NumPy array
faces = np.asarray(faces)
# Reshape the array to have 100 rows (one for each face image) and -1 columns (automatically determine the number of columns)
faces = faces.reshape(100, -1)

# Save the child's name and face data to files
if 'names.pkl' not in os.listdir('data/'):
    # If the names file doesn't exist, create it with the child's name repeated 100 times
    names = [child_name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # If the names file already exists, load it and append the child's name to the existing list
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [child_name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save the face data to a file
if 'faces_data.pkl' not in os.listdir('data/'):
    # If the face data file doesn't exist, create it with the captured face data
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
else:
    # If the face data file already exists, load it and append the captured face data to the existing array
    with open('data/faces_data.pkl', 'rb') as f:
        existing_faces = pickle.load(f)
    faces = np.append(existing_faces, faces, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)