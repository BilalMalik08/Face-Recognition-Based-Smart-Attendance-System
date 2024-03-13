# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import win32com.client

# Initialize the text-to-speech engine
engine = win32com.client.Dispatch("SAPI.SpVoice")

# Open the webcam
cam = cv2.VideoCapture(0)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Check if the files exist
names_file_path = 'data/names.pkl'
faces_file_path = 'data/faces_data.pkl'

if os.path.exists(names_file_path) and os.path.exists(faces_file_path):
    # Load labels and faces data
    with open(names_file_path, 'rb') as f:
        LABELS = pickle.load(f)

    with open(faces_file_path, 'rb') as f:
        FACES = pickle.load(f)

    # Flatten and reshape faces data
    FACES = FACES.reshape(len(FACES), -1)  # Flatten each image
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    # Load the background image
    image_bg = cv2.imread("background.png")

    Col_Names = ["Name", "Date", "Time"]

    # Loop to capture face images
    while True:
        # Read a frame from the webcam
        ret, frame = cam.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Dictionary to store attendance records for each face
        attendance_records = {}

        # Loop through each detected face
        for (x, y, w, h) in detected_faces:
            # Crop the detected face region
            face_img = frame[y:y + h, x:x + w, :]

            # Resize the cropped face image to a standard size
            resized_face = cv2.resize(face_img, (50, 50))

            # Flatten and reshape the resized face image
            resized_face = resized_face.flatten().reshape(1, -1)

            # Predict label for the face
            output = knn.predict(resized_face)

            # Get current date and time
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            timeStamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            # Check if the attendance file for the current date exists
            file_exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

            # Display the predicted label in green color
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            # Create an attendance record for the current face
            attendance = [str(output[0]), date, timeStamp]

            # Add the attendance record to the dictionary with face coordinates as key
            attendance_records[(x, y, w, h)] = attendance

        # Assign the frame to the background image
        image_bg[162: 162 + 480, 55: 55 + 640] = frame

        # Display the frame with face detection
        cv2.imshow("Frame", image_bg)

        # Check for key press events
        key = cv2.waitKey(1)

        # If the user presses 'o', write the attendance records to a CSV file
        if key == ord('o'):
            # Speak "Attendance Taken"
            engine.Speak("Attendance Taken")
            time.sleep(5)

            if file_exist:
                with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    # Write attendance records from the dictionary
                    for attendance_record in attendance_records.values():
                        writer.writerow(attendance_record)
            else:
                with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(Col_Names)
                    # Write attendance records from the dictionary
                    for attendance_record in attendance_records.values():
                        writer.writerow(attendance_record)

        # If the user presses 'q', break out of the loop
        if key == ord('q'):
            # Speak "Exit"
            engine.Speak("Exit")
            break

    # Release the webcam
    cam.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Files not found. Make sure 'names.pkl' and 'faces_data.pkl' exist in the 'data' directory.")