import cv2
import mediapipe as mp
import math
import time
import numpy as np
import subprocess
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'hand_landmarker.task' 

WRIST      = 0
THUMB_TIP  = 4
INDEX_TIP  = 8
INDEX_MCP  = 5
INDEX_PIP  = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
MIDDLE_MCP = 9
RING_TIP   = 16
RING_PIP   = 14
RING_MCP   = 13
PINKY_TIP  = 20
PINKY_PIP  = 18
PINKY_MCP  = 17

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_finger_Extend(lm, tip, pip):
    wrist = lm[WRIST]
    return dist(wrist, lm[tip]) > dist(wrist, lm[pip]) * 1.05

def is_finger_curled(lm, tip, pip):
    wrist = lm[WRIST]
    return dist(wrist, lm[tip]) < dist(wrist, lm[pip])

def check_heart(lm):
    index_up  = is_finger_Extend(lm, INDEX_TIP, INDEX_PIP)
    thumb_up  = is_finger_Extend(lm, THUMB_TIP, INDEX_MCP)
    middle_dn = is_finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_dn   = is_finger_curled(lm, RING_TIP, RING_PIP)
    pinky_dn  = is_finger_curled(lm, PINKY_TIP, PINKY_PIP)
    return index_up and thumb_up and middle_dn and ring_dn and pinky_dn

def check_two_heart(hand_lms_list, index_thresh=0.25, thumb_thresh=0.25):
    if len(hand_lms_list) < 2:
        return False
    lm0, lm1 = hand_lms_list[0], hand_lms_list[1]
    if not (check_heart(lm0) and check_heart(lm1)):
        return False
    
    idx_dist = dist(lm0[INDEX_TIP], lm1[INDEX_TIP])
    thumb_dist = dist(lm0[THUMB_TIP], lm1[THUMB_TIP])
    
    tips_close = idx_dist < index_thresh
    thumbs_close = thumb_dist < thumb_thresh
    
    avg_idx_y = (lm0[INDEX_TIP].y + lm1[INDEX_TIP].y) / 2
    avg_thumb_y = (lm0[THUMB_TIP].y + lm1[THUMB_TIP].y) / 2
    
    return tips_close and thumbs_close and (avg_idx_y < avg_thumb_y)

def draw_heart_overlay(frame, cx, cy, size=100, alpha=0.6):
    overlay = frame.copy()
    r = size // 3
    cv2.circle(overlay, (cx - r, cy - r // 2), r, (0, 0, 220), -1)
    cv2.circle(overlay, (cx + r, cy - r // 2), r, (0, 0, 220), -1)
    pts = [(cx - size // 2, cy - r // 3), (cx + size // 2, cy - r // 3), (cx, cy + size // 2)]
    cv2.fillPoly(overlay, [np.array(pts)], (0, 0, 220))
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def main():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    heart_active = False
    last_heart_ts = 0
    HOLD_DURATION = 0.5

    with vision.HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            frame_timestamp_ms = int(time.time() * 1000)
            result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            all_lms = []
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    all_lms.append(hand_landmarks)

            is_heart = check_two_heart(all_lms)

            if is_heart:
                if not heart_active:
                    subprocess.Popen(["start", r"C:\Users\kirat\Downloads\so-easy-to-fall-in-love---olivia-dean.mp4"],shell=True)
                heart_active = True
                last_heart_ts = time.time()
            elif heart_active and (time.time() - last_heart_ts > HOLD_DURATION):
                heart_active = False

            if heart_active and len(all_lms) == 2:
                cx = int((all_lms[0][INDEX_TIP].x + all_lms[1][INDEX_TIP].x + 
                          all_lms[0][THUMB_TIP].x + all_lms[1][THUMB_TIP].x) / 4 * w)
                cy = int((all_lms[0][INDEX_TIP].y + all_lms[1][INDEX_TIP].y + 
                          all_lms[0][THUMB_TIP].y + all_lms[1][THUMB_TIP].y) / 4 * h)
                draw_heart_overlay(frame, cx, cy)
                cv2.putText(frame, "HEART DETECTED", (w // 2 - 140, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Form a heart with both hands", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('MediaPipe Tasks Heart Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()