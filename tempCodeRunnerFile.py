import cv2
import pickle
import numpy as np
import os

# Open the webcam
cam = cv2.VideoCapture(0)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Initialize empty lists for storing face images and names
faces = []
names = []

# Counter for the number of captured face images
counter = 0

# Get the user's name
user_name = input("Enter your name: ")

# Loop to capture face images
while True:
    # Read a frame from the webcam
    ret, frame = cam.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through each detected face
    for (x, y, w, h) in detected_faces:
        # Crop the detected face region
        face_img = frame[y:y+h, x:x+w, :]
        
        # Resize the cropped face image to a standard size
        resized_face = cv2.resize(face_img, (50, 50))
        
        # Add the resized face image to the list
        if counter % 10 == 0:
            faces.append(resized_face)
            names.append(user_name)
        counter += 1
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        
        # Display the count of captured face images
        cv2.putText(frame, f"Faces Captured: {len(faces)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with face detection
    cv2.imshow("Frame", frame)
    
    # Check for key press events
    key = cv2.waitKey(1)
    
    # If the user presses 'q' or the maximum number of face images is captured, break out of the loop
    if key == ord('q') or len(faces) == 100:
        break

# Release the webcam
cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Convert the lists of face images and names to NumPy arrays
faces = np.asarray(faces)
names = np.asarray(names)

# Save the user's name and face data to files
with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces, f)

# Check if the files exist
if os.path.exists('data/names.pkl') and os.path.exists('data/faces_data.pkl'):
    # Load the names
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    
    # Load the face data
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
else:
    print("No existing data found.")