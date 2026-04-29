# Y1S2-Final-Code
This code is used for line tracking while detecting symbols or arrows simultaneously.

import threading
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from collections import deque


#1) Shared State
class SharedState:
    def __init__(self):
        self.frame = None
        self.debug_frame = None
        self.running = True
        self.lock = threading.Lock()

        self.line_cx = None
        self.active_symbol = "None"
        self.active_arrow = "None"
        self.current_color = "None"

        self.last_dir = 0


state = SharedState()

#2) GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

PINS = {"L": (17, 18, 27), "R": (23, 24, 22)}
for side in PINS.values():
    for pin in side: GPIO.setup(pin, GPIO.OUT)

pwm_l = GPIO.PWM(27, 800);
pwm_r = GPIO.PWM(22, 800)
pwm_l.start(0);
pwm_r.start(0)


def set_motors(l, r, mode="FORWARD"):
    l, r = max(0, min(80, float(l))), max(0, min(80, float(r)))
    if mode == "FORWARD":
        GPIO.output(17, 1);
        GPIO.output(18, 0)
        GPIO.output(23, 1);
        GPIO.output(24, 0)
    elif mode == "REVERSE_LEFT":
        GPIO.output(17, 0);
        GPIO.output(18, 1)
        GPIO.output(23, 1);
        GPIO.output(24, 0)
    elif mode == "REVERSE_RIGHT":
        GPIO.output(17, 1);
        GPIO.output(18, 0)
        GPIO.output(23, 0);
        GPIO.output(24, 1)
    pwm_l.ChangeDutyCycle(l);
    pwm_r.ChangeDutyCycle(r)


#3) Camera Thread
def thread_camera():
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"format": "XRGB8888", "size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({"AnalogueGain": 4.0})
    picam2.set_controls({"ExposureTime": 20000})

    while state.running:
        frame = picam2.capture_array()
        with state.lock:
            state.frame = frame
    picam2.stop()


#4) Vision Thread
def thread_vision():
    sym_buffer = deque(maxlen=5)
    fast_buffer = deque(maxlen=2)

    while state.running:
        with state.lock:
            if state.frame is None: continue
            img = state.frame.copy()

        debug_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #A: LINE DETECTION (Priority: Red > Yellow > Black)
        roi = img[170:240, 0:320]
        hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)

        found_cx, col_name, col_bgr = None, "None", (125, 125, 125)

        r_low = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        r_high = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))

        masks = [
            (cv2.bitwise_or(r_low, r_high), "RED", (0, 0, 255)),
            (cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255])), "YELLOW", (0, 255, 255)),
            (cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75])), "BLACK", (0, 0, 0))
        ]

        for mask, name, bgr in masks:
            M = cv2.moments(mask)
            if M["m00"] > 500:
                found_cx = int(M["m10"] / M["m00"])
                col_name, col_bgr = name, bgr
                cv2.circle(debug_img, (found_cx, 205), 10, col_bgr, -1)
                break

        #B: SYMBOL DETECTION
        sym_roi = img[0:170, 0:320]
        hsv_sym = cv2.cvtColor(sym_roi, cv2.COLOR_BGR2HSV)

        sym_gray = cv2.cvtColor(sym_roi, cv2.COLOR_BGR2GRAY)
        sym_blur = cv2.GaussianBlur(sym_gray, (5, 5), 0)
        sym_thresh = cv2.adaptiveThreshold(sym_blur, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7)
        sym_kernel = np.ones((3, 3), np.uint8)
        sym_thresh = cv2.erode(sym_thresh, sym_kernel, iterations=1)
        sym_thresh = cv2.dilate(sym_thresh, sym_kernel, iterations=1)

        hazard_cnts, hazard_hier = cv2.findContours(sym_thresh,
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        color_sym = "None"
        recycle_sym = "None"

        #Hazard: circle + nested triangle + yellow background
        if hazard_hier is not None:
            hier = hazard_hier[0]
            found_circle_idx = -1
            found_triangle_inside = False

            for i, cnt in enumerate(hazard_cnts):
                area = cv2.contourArea(cnt)
                if area < 1000:
                    continue

                approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1e-5)

                if len(approx) > 6 and solidity > 0.70 and aspect < 1.5:
                    (cx_c, cy_c), radius = cv2.minEnclosingCircle(cnt)
                    circle_area = np.pi * radius * radius
                    circularity = area / (circle_area + 1e-5)
                    if circularity > 0.55 and area > 2000:
                        found_circle_idx = i

                if len(approx) == 3 and area > 300 and hier[i][3] != -1:
                    found_triangle_inside = True

            if found_circle_idx != -1 and found_triangle_inside:
                yellow_bg = cv2.inRange(hsv_sym,
                                        np.array([18, 80, 80]),
                                        np.array([38, 255, 255]))
                if cv2.countNonZero(yellow_bg) > 200:
                    color_sym = "Hazard"

        #Button Code & Recycle: Green Pixel Density
        total_pixels = sym_roi.shape[0] * sym_roi.shape[1]

        green_mask = cv2.inRange(hsv_sym,
                                 np.array([52, 43, 33]),
                                 np.array([105, 255, 255]))

        green_ratio = cv2.countNonZero(green_mask) / (total_pixels + 1e-5)

        green_button = "None"

        if green_ratio > 0.25:
            green_button = "Button Code"
        else:
            recycle_mask = cv2.inRange(hsv_sym,
                                       np.array([67, 71, 55]),
                                       np.array([100, 255, 255]))
            recycle_pixels = cv2.countNonZero(recycle_mask)

            if recycle_pixels > 500:
                points = cv2.findNonZero(recycle_mask)
                if points is not None:
                    rx, ry, rw, rh = cv2.boundingRect(points)
                    if rw < 160 and rh < 160:
                        cx = rx + (rw // 2)
                        cy = ry + (rh // 2)
                        center_slice = recycle_mask[
                                       max(0, cy - 5):cy + 5,
                                       max(0, cx - 5):cx + 5]
                        center_pixels = cv2.countNonZero(center_slice)
                        if center_pixels < 5.5:
                            recycle_sym = "Recycle"

        #QR Code: Dark navy blue detected
        navy_mask = cv2.inRange(hsv_sym,
                                np.array([100, 130, 50]),
                                np.array([125, 255, 160]))
        navy_area = cv2.countNonZero(navy_mask)

        qr_sym = "None"
        if navy_area > 2500:
            pts = cv2.findNonZero(navy_mask)
            if pts is not None:
                qx, qy, qw, qh = cv2.boundingRect(pts)
                density = navy_area / (qw * qh + 1e-5)
                if density > 0.31:
                    qr_sym = "QR Code"

        #Fingerprint: Purple detected
        purple_mask = cv2.inRange(hsv_sym,
                                  np.array([128, 60, 60]),
                                  np.array([152, 255, 255]))
        purple_area = cv2.countNonZero(purple_mask)
        fingerprint_sym = "Fingerprint" if purple_area > 1000 and navy_area < 1500 else "None"

        #Symbol Priority
        if color_sym != "None":
            active_sym_raw = color_sym
        elif green_button != "None":
            active_sym_raw = green_button
        elif recycle_sym != "None":
            active_sym_raw = recycle_sym
        elif fingerprint_sym != "None":
            active_sym_raw = fingerprint_sym
        elif qr_sym != "None":
            active_sym_raw = qr_sym
        else:
            active_sym_raw = "None"

        fast_buffer.append(active_sym_raw)
        sym_buffer.append(active_sym_raw)
        active_sym = active_sym_raw if fast_buffer.count(active_sym_raw) >= 2 else "None"

        #C: ARROW DETECTION
        active_arrow = "None"
        if active_sym == "None":
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 51, 7)

            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)

                if 1500 < area < 8000:
                    approx = cv2.approxPolyDP(cnt,
                                              0.03 * cv2.arcLength(cnt, True), True)

                    if 6 <= len(approx) <= 9:
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 1.0

                        if 0.5 < solidity < 0.85:
                            x, y, w, h = cv2.boundingRect(cnt)

                            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                            if aspect_ratio > 3.5:
                                continue

                            roi_arrow = img[y:y + h, x:x + w]
                            hsv_arrow = cv2.cvtColor(roi_arrow, cv2.COLOR_BGR2HSV)
                            roi_pixels = w * h

                            # (green, blue, red, orange) with lower pixel thresholds
                            green_px = cv2.countNonZero(cv2.inRange(hsv_arrow,
                                                                    np.array([40, 80, 80]), np.array([85, 255, 255])))
                            blue_px = cv2.countNonZero(cv2.inRange(hsv_arrow,
                                                                   np.array([95, 80, 80]), np.array([130, 255, 255])))
                            red_px_lo = cv2.countNonZero(cv2.inRange(hsv_arrow,
                                                                     np.array([0, 100, 100]), np.array([10, 255, 255])))
                            red_px_hi = cv2.countNonZero(cv2.inRange(hsv_arrow,
                                                                     np.array([160, 100, 100]),
                                                                     np.array([180, 255, 255])))
                            red_px = red_px_lo + red_px_hi
                            orange_px = cv2.countNonZero(cv2.inRange(hsv_arrow,
                                                                     np.array([10, 150, 150]),
                                                                     np.array([25, 255, 255])))

                            max_color_px = max(green_px, blue_px, red_px, orange_px)

                            color_ratio = max_color_px / (roi_pixels + 1e-5)
                            is_colored_arrow = color_ratio > 0.15 or max_color_px > 150

                            if not is_colored_arrow:
                                continue

                            # Confirm arrow notch via convexity defects
                            has_notch = False
                            hull_idx = cv2.convexHull(cnt, returnPoints=False)
                            if hull_idx is not None and len(hull_idx) >= 3:
                                try:
                                    defects = cv2.convexityDefects(cnt, hull_idx)
                                    if defects is not None:
                                        deep = [d for d in defects[:, 0]
                                                if d[3] / 256.0 > 6]
                                        has_notch = len(deep) >= 1
                                except:
                                    pass

                            if not has_notch:
                                continue

                            # Determine direction from centroid
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                bX = x + w / 2
                                bY = y + h / 2

                                dx = abs(cX - bX)
                                dy = abs(cY - bY)

                                if dx > dy:
                                    # Horizontal arrow
                                    raw_dir = "Arrow Right" if cX > bX else "Arrow Left"
                                else:
                                    # Vertical arrow, only Up matters
                                    raw_dir = "Arrow Up" if cY < bY else "Arrow Down"

                                if raw_dir in ("Arrow Up", "Arrow Left", "Arrow Right"):
                                    active_arrow = raw_dir
                                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 2)
                            break

        #Heads Up Display
        cv2.rectangle(debug_img, (0, 0), (190, 100), (40, 40, 40), -1)
        cv2.putText(debug_img, f"LINE: {col_name}", (10, 25), 1, 1.2, col_bgr, 2)
        cv2.putText(debug_img, f"SYM: {active_sym}", (10, 55), 1, 1.2, (255, 255, 255), 2)
        cv2.putText(debug_img, f"ARR: {active_arrow}", (10, 85), 1, 1.2, (0, 255, 0), 2)
        cv2.line(debug_img, (0, 170), (320, 170), (255, 255, 0), 1)

        with state.lock:
            state.line_cx = found_cx
            state.current_color = col_name
            state.active_symbol = active_sym
            state.active_arrow = active_arrow
            state.debug_frame = debug_img

        time.sleep(0.04)


#5: Motor Control Thread with PID
def thread_motor():
    Kp, Ki, Kd = 0.33, 0.005, 0.28
    integral, last_err = 0, 0
    base = 35

    while state.running:
        with state.lock:
            cx, sym, arrow = state.line_cx, state.active_symbol, state.active_arrow

        #A: Symbols
        if sym in ["Hazard", "Button Code"]:
            set_motors(0, 0);
            time.sleep(2.0)
            set_motors(45, 45);
            time.sleep(0.4)
            with state.lock:
                state.active_symbol = "None"
            continue

        elif sym == "Recycle":
            set_motors(60, 60, "REVERSE_RIGHT");
            time.sleep(4.2)
            set_motors(45, 45, "FORWARD");
            time.sleep(0.2)
            with state.lock:
                state.active_symbol = "None"
            continue

        elif sym in ["QR Code", "Fingerprint"]:
            print("Biometric Logged")
            with state.lock:
                state.active_symbol = "None"
            continue

        #B: Arrow direction movement
        elif arrow != "None":
            if arrow == "Arrow Right":
                set_motors(40, 0);  time.sleep(0.6)
            elif arrow == "Arrow Left":
                set_motors(0, 40); time.sleep(0.6)
            elif arrow == "Arrow Up":
                set_motors(30, 30); time.sleep(0.6)
            with state.lock:
                state.active_arrow = "None"

        #C: Line PID
        elif cx is not None:
            err = 160 - cx
            integral = max(-500, min(500, integral + err))
            turn = (Kp * err) + (Ki * integral) + (Kd * (err - last_err))
            set_motors(base - turn, base + turn)
            last_err = err
            if err < -40:
                set_motors(25 + abs(turn), 25 + abs(turn), "REVERSE_RIGHT")
            elif err > 40:
                set_motors(25 + abs(turn), 25 + abs(turn), "REVERSE_LEFT")
            state.last_dir = 1 if err < -80 else -1 if err > 80 else 0
        else:
            if state.last_dir == 1:
                set_motors(45, 0, "FORWARD")
            elif state.last_dir == -1:
                set_motors(0, 45, "FORWARD")

        time.sleep(0.01)


#6: Main loop
if __name__ == "__main__":
    t1 = threading.Thread(target=thread_camera)
    t2 = threading.Thread(target=thread_vision)
    t3 = threading.Thread(target=thread_motor)

    for t in [t1, t2, t3]: t.start()

    try:
        while True:
            with state.lock:
                if state.debug_frame is not None:
                    cv2.imshow("Robot Master Debug", state.debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        state.running = False
        time.sleep(0.5)
        set_motors(0, 0)
        GPIO.cleanup()
        cv2.destroyAllWindows()
