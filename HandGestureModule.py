import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = staticMode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    # Find and connect the landmarks of the hands.
    # para: img, connect - bool
    # return: img
    def findHands(self, img, connect=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        # Handling the landmarks (value extract from the hand tracking)
        if self.result.multi_hand_landmarks:
            for handLandmarks in self.result.multi_hand_landmarks:

                # If connect is True, draw the connections between the dots
                if connect:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNumber=0, connect=True):

        landmarkList = []
        if self.result.multi_hand_landmarks:
            currentHand = self.result.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(currentHand.landmark):
                # Getting the Pixel value of the landmark instead of decimal
                height, width, channels = img.shape

                # Getting the location of the landmark starting from the center of the screen
                centerX, centerY = int(landmark.x*width), int(landmark.y*height)

                # Append the id and location of landmarks to the list
                landmarkList.append([id, centerX, centerY])

                # Drawing a circle at the landmark with id == 4
                # para: img, location to draw, size, color, thickness
                if connect and id == 4:
                    cv2.circle(img, (centerX, centerY), 10, (0, 255, 255), -1)

        return landmarkList


def main():
    previousTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()

        # Set img as returned from findHand()
        img = detector.findHands(img)

        # Set landmarkList as return from findPosition()
        # Check for hand then print landmark id 4
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])

        # Calculate the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Flip the img to get a mirrored view from the webcam
        flip = cv2.flip(img, 1)

        # Print the FPS on the flip img
        # para: img, string, distant from the top-left-corner, font, size, color, thickness
        cv2.putText(flip, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # Display the image
        cv2.imshow("Flip img", flip)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
