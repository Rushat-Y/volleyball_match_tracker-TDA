import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def identify_move(hand_landmarks):
    lm = hand_landmarks.landmark
    fingertips = [lm[i].y for i in [8, 12, 16, 20]]
    lower_joints = [lm[i - 2].y for i in [8, 12, 16, 20]]

    if all(fingertips[i] > lower_joints[i] for i in range(4)):
        return "rock"
    elif all(fingertips[i] < lower_joints[i] for i in range(4)):
        return "paper"
    elif fingertips[0] < lower_joints[0] and fingertips[1] < lower_joints[1] and \
         fingertips[2] > lower_joints[2] and fingertips[3] > lower_joints[3]:
        return "scissors"
    return "unrecognized"

camera = cv.VideoCapture(0)
timer = 0
p1_choice = p2_choice = None
message = ""
valid = True

with mp_hands.Hands(model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands_model:

    while True:
        ret, img = camera.read()
        img = cv.resize(img, (1280, 720))

        if not ret or img is None:
            break

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = hands_model.process(img_rgb)
        img = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        img = cv.flip(img, 1)

        # Game clock 
        if 0 <= timer < 20:
            valid = True
            message = "Get Ready"
        elif timer < 30:
            message = "3..."
        elif timer < 50:
            message = "2..."
        elif timer < 60:
            message = "1... Show!"
        elif timer == 60:
            hand_list = result.multi_hand_landmarks
            if hand_list and len(hand_list) == 2:
                p1_choice = identify_move(hand_list[0])
                p2_choice = identify_move(hand_list[1])
            else:
                valid = False
        elif timer < 100:
            if valid:
                message = f"Player 1: {p1_choice}, Player 2: {p2_choice}."
                if p1_choice == p2_choice:
                    message += " It's a tie!"
                elif (p1_choice, p2_choice) in [("paper", "rock"), ("rock", "scissors"), ("scissors", "paper")]:
                    message += " Player 1 wins!"
                else:
                    message += " Player 2 wins!"
            else:
                message = "Improper move detected!"

        cv.putText(img, f"Timer: {timer}", (50, 50), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv.putText(img, message, (50, 80), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv.putText(img, "Press 'Q' to Exit", (50, 120), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        timer = (timer + 1) % 100
        cv.imshow('Rock Paper Scissors Game', img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print("Quitting the game...")
            break

camera.release()
cv.destroyAllWindows()
