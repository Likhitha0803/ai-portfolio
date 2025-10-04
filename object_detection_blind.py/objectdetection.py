import cv2
import torch
import pyttsx3

# Load pre-trained YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speaking speed

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

spoken_labels = set()  # to avoid repeating continuously

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    results = model(frame)

    # Extract labels
    detected = results.pred[0]
    labels = results.names

    current_labels = set()

    for *box, conf, cls in detected:
        label = labels[int(cls)]
        current_labels.add(label)

        # Draw bounding boxes
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Speak new detections only
    for label in current_labels:
        if label not in spoken_labels:
            engine.say(f"{label} detected")
            engine.runAndWait()

    spoken_labels = current_labels  # update for next frame

    # Show video feed with boxes
    cv2.imshow("Object Detection - Blind Assist", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
