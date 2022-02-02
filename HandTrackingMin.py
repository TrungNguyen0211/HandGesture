import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # Handling the landmarks (value extract from the hand tracking)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            for id, landmark in enumerate(handLms.landmark):
                # Getting the Pixel value of the landmark instead of decimal
                height, width, channels = img.shape

                # Getting the location of the landmark starting from the center of the screen
                centerX, centerY = int(landmark.x*width), int(landmark.y*height)
                print(id, centerX, centerY)

                # Drawing a circle at the landmark with id == 4
                # para: img, location to draw, size, color, thickness
                if id == 4:
                    cv2.circle(img, (centerX, centerY), 10, (0, 255, 255), cv2.FILLED)
            # Create connections between the dots
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the FPS
    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    # Flip the img to get a mirrored view from the webcam
    flip = cv2.flip(img, 1)

    # Print the FPS on the flip img
    # para: img, string, distant from the top-left-corner, font, size, color, thickness
    cv2.putText(flip, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)

    # Display the image
    cv2.imshow("Flip img", flip)
    cv2.waitKey(1)