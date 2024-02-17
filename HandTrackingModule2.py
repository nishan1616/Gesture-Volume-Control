import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode, max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_con,
                                         min_tracking_confidence=self.tracking_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_ids = [4,8,12,16,20]

    def draw_hands(self, img, draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_landmarks(self, img, hand_num = 0, draw= True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lm_list.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,255),cv2.FILLED)
        return self.lm_list


def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.draw_hands(img)
        lmlist = detector.find_landmarks(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()