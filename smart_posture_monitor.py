import cv2
import mediapipe as mp
import time
import winsound
import math

# ---------------- SETTINGS ---------------- #

SHOW_ANALYSIS = False   # Toggle with 'L'

# Soft colors (BGR)
GREEN = (80, 200, 120)
RED = (100, 100, 255)
YELLOW = (120, 220, 220)
WHITE = (240, 240, 240)
BLUE = (200, 180, 120)

# ---------------- FUNCTIONS ---------------- #

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x)
    )
    return abs(angle)

# ---------------- INITIAL SETUP ---------------- #

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

start_time = time.time()

bad_posture_start = None
bad_count = 0
strict_warnings = 0

bad_events = []
warning_display_start = None

# ---------------- MAIN LOOP ---------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = pose.process(rgb)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks.landmark

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        nose = landmarks[0]

        # ---------------- POSTURE LOGIC ---------------- #

        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2

        posture = "Good"
        posture_score = 100

        # Detect left + right + forward slouch
        if abs(nose.x - shoulder_x) > 0.03 or nose.y > left_shoulder.y + 0.05:

            posture = "Bad"
            posture_score = 50
            bad_count += 1

            if bad_posture_start is None:
                bad_posture_start = time.time()

            else:
                if bad_count > 5:

                    strict_warnings += 1
                    warning_time = int(time.time() - start_time)
                    bad_events.append(warning_time)

                    warning_display_start = time.time()

                    winsound.Beep(1200, 400)

                    bad_count = 0
                    bad_posture_start = None

        else:
            bad_count = 0
            bad_posture_start = None

        # ---------------- DISTANCE CHECK ---------------- #

        face_distance = abs(nose.y - left_shoulder.y)

        if face_distance > 0.45:
            cv2.putText(frame, "Too Close to Screen",
                        (450, 650),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        YELLOW, 2)

        # ---------------- WARNING DISPLAY ---------------- #

        if warning_display_start is not None:
            if time.time() - warning_display_start < 5:
                cv2.putText(frame, "⚠ Maintain Proper Posture",
                            (350, 650),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            YELLOW, 3)
            else:
                warning_display_start = None

        # ---------------- ANALYSIS MODE ---------------- #

        if SHOW_ANALYSIS:

            left_eye = landmarks[2]
            right_eye = landmarks[5]

            # draw points
            for point in [nose, left_shoulder, right_shoulder, left_eye, right_eye]:
                cx, cy = int(point.x * 1280), int(point.y * 720)
                cv2.circle(frame, (cx, cy), 6, WHITE, -1)

            # draw line
            cv2.line(frame,
                     (int(nose.x * 1280), int(nose.y * 720)),
                     (int(left_shoulder.x * 1280), int(left_shoulder.y * 720)),
                     WHITE, 2)

            # angle calculation
            angle = calculate_angle(left_shoulder, nose, right_shoulder)

            cv2.putText(frame, f"Angle: {int(angle)}",
                        (550, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        WHITE, 2)

        # ---------------- DISPLAY ---------------- #

        elapsed_time = int(time.time() - start_time)

        # Top Left: Time
        cv2.putText(frame, f"Time: {elapsed_time}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    WHITE, 2)

        # Top Right: Posture
        cv2.putText(frame, f"Posture: {posture}",
                    (900, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    GREEN if posture == "Good" else RED, 2)

        # Center: Score
        cv2.putText(frame, f"Score: {posture_score}%",
                    (550, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    BLUE, 2)

    else:
        cv2.putText(frame, "User Not Detected",
                    (450, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    RED, 2)

    cv2.imshow("Smart Posture Monitoring System", frame)

    # ---------------- KEY CONTROLS ---------------- #

    key = cv2.waitKey(1) & 0xFF

    if key == ord('l'):
        SHOW_ANALYSIS = not SHOW_ANALYSIS

    if key == 27:
        break


# ---------------- SUMMARY ---------------- #

cap.release()
cv2.destroyAllWindows()

total_time = int(time.time() - start_time)

print("\n------ SESSION SUMMARY ------")
print(f"Total Time: {total_time} sec")
print(f"Strict Warnings: {strict_warnings}")
print("Warning triggered at (seconds):")
print(bad_events)
