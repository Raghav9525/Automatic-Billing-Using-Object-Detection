import math
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure you have this module or adjust to your specific YOLO version
from utils import config  # Ensure this module contains the required 'classNames'
import cvzone

def detections():
    # Load class names
    classNames = config.classNames

    # Load detector
    detector = YOLO('./YOLOweights/yolov8l.pt')  # Adjust to your specific weights and model version

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Temporary storage for this frame's IDs
    frame_ids = {
        "bananaCount": [],
        "appleCount": [],
        "pizzaCount": []
    }

    try:
        while True:
            ret, frame = cap.read()
            outputs = detector(frame, stream=True)

            bananaCount = 0
            appleCount = 0
            pizzaCount = 0
            for output in outputs:
                boxes = output.boxes
                for box in boxes:
                    className = classNames[int(box.cls[0])]

                    if className =="banana" or className =="apple" or className =="pizza":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 0))
                        cvzone.putTextRect(frame, f'{className}', (x1, y1 - 10), scale=1, thickness=2, offset=0)

                        if className == "banana":
                            bananaCount += 1
                        elif className == "apple":
                            appleCount += 1
                        elif className == "pizza":
                            pizzaCount += 1

            print(f"Banana: {bananaCount}, Apple: {appleCount}, Pizza: {pizzaCount}")

            # Append counts to the respective keys in frame_ids
            frame_ids["bananaCount"].append(bananaCount)
            frame_ids["appleCount"].append(appleCount)
            frame_ids["pizzaCount"].append(pizzaCount)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # This block will execute when Ctrl+C is pressed
        print("Program interrupted.")

    finally:
        # Iterate over the items in frame_ids
        for item, counts in frame_ids.items():
            if counts:  # This check ensures the list is not empty
                max_count = max(counts)
                print(f"Maximum {item}: {max_count}")
            else:
                print(f"Maximum {item}: No data")

        cap.release()
        cv2.destroyAllWindows()

# Now you can call the function to start the detection
# detections()
