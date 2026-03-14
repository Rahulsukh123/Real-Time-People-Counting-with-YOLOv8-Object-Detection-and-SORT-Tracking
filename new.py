import cv2
import numpy as np
import pyttsx3

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to announce the finger count
def announce_finger_count(count):
    engine.say(f'Fingers: {count}')
    engine.runAndWait()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for hand detection
    roi = frame[100:300, 100:300]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # Create a bounding rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the convex hull and convexity defects
        hull = cv2.convexHull(cnt)
        hull2 = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull2)

        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                # Calculate the angle between the start, end, and far points
                a = np.linalg.norm(np.array(start) - np.array(far))
                b = np.linalg.norm(np.array(end) - np.array(far))
                c = np.linalg.norm(np.array(start) - np.array(end))
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # If the angle is less than 90 degrees, consider it as a finger
                if angle <= 90:
                    count += 1
                    cv2.circle(roi, far, 4, [0, 0, 255], -1)

            # Display the number of fingers
            finger_count = count + 1
            cv2.putText(frame, f'Fingers: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Announce the finger count
            announce_finger_count(finger_count)

    # Display the frame with the detected fingers
    cv2.imshow('Frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()