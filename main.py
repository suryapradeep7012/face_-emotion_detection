import cv2
from transformers import pipeline
from PIL import Image

# Load Hugging Face model
emotion_pipeline = pipeline("image-classification", model="trpakov/vit-face-expression", framework="pt")

# Webcam
cap = cv2.VideoCapture(0)
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Crop face (convert OpenCV BGR -> RGB -> PIL)
        roi = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)

        # Run prediction with Hugging Face model
        preds = emotion_pipeline(pil_img)

        # Get top prediction
        label = preds[0]["label"]
        score = preds[0]["score"]

        # Put label
        cv2.putText(frame, f"{label} ({score:.2f})", (x, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Video", cv2.resize(frame, (1600,960), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
