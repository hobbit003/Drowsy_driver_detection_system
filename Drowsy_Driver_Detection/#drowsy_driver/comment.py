# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

# Initialize Pygame mixer for sound
mixer.init()
# Load alert sound
mixer.music.load("new.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set thresholds and parameters
thresh = 0.25
frame_check = 20

# Initialize dlib face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define indices for left and right eyes from facial landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Open video capture from default camera (0)
cap = cv2.VideoCapture(0)

# Initialize flag and countdown variables
flag = 0
countdown_start_time = time.time()
countdown_active = False

# Main loop to process each frame
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    # Resize frame for faster processing
    frame = imutils.resize(frame, width=450)
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    subjects = detect(gray, 0)

    # If no faces are detected
    if len(subjects) == 0:
        # If countdown is not active, start the countdown
        if not countdown_active:
            countdown_start_time = time.time()
            countdown_active = True
        else:
            # Calculate time elapsed without face
            elapsed_time_without_face = time.time() - countdown_start_time
            # Calculate remaining countdown display
            countdown_display = int(4 - elapsed_time_without_face)
            # If still within the countdown duration, display countdown
            if elapsed_time_without_face < 4:
                cv2.putText(frame, f"Countdown: {countdown_display}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # If countdown is complete, issue alert for no face detected and play sound
                cv2.putText(frame, "*************ALERT! No Face Detected!************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
    else:
        # If faces are detected, reset the countdown
        countdown_active = False

        # Loop through detected faces
        for subject in subjects:
            # Predict facial landmarks
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            # Extract left and right eye regions
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # If either left or right eye is not detected
            if len(leftEye) == 0 or len(rightEye) == 0:
                # If countdown is not active, start the countdown
                if not countdown_active:
                    countdown_start_time = time.time()
                    countdown_active = True
                else:
                    # Calculate time elapsed without eyes
                    elapsed_time_without_eyes = time.time() - countdown_start_time
                    # Calculate remaining countdown display
                    countdown_display = int(4 - elapsed_time_without_eyes)
                    # If still within the countdown duration, display countdown
                    if elapsed_time_without_eyes < 4:
                        cv2.putText(frame, f"Countdown: {countdown_display}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        # If countdown is complete, issue alert for eyes not detected and play sound
                        cv2.putText(frame, "****************ALERT! Eyes Not Detected!****************", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
            else:
                # If eyes are detected, reset the countdown
                countdown_active = False

                # Calculate Eye Aspect Ratio (EAR)
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                # Convex hulls for eye regions
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)

                # Draw rectangle around the detected face
                (x, y, w, h) = face_utils.rect_to_bb(subject)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Draw contours around the eyes
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # If EAR falls below the threshold, increment the flag counter
                if ear < thresh:
                    flag += 1
                    print(flag)
                    # If the flag counter exceeds the frame check, issue drowsiness alert and play sound
                    if flag >= frame_check:
                        cv2.putText(frame, "**********ALERT! Drowsiness Detected!***********", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "**********ALERT! Drowsiness Detected!***********", (10, 325),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                else:
                    # If EAR is above the threshold, reset the flag counter
                    flag = 0

    # Display the frame with visual indicators
    cv2.imshow("Frame", frame)
    # Wait for user input to exit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Close all OpenCV windows and release video capture resources
cv2.destroyAllWindows()
cap.release()
