from sign_language_db import insert_prediction
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# ---- Webcam ----
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# ---- Load 3 Models ----
classifier1 = Classifier("Models/numbers_model.h5", "Models/numbers_labels.txt")
classifier2 = Classifier("Models/alphabet_model.h5", "Models/alphabet_labels.txt")
classifier3 = Classifier("Models/words_model.h5", "Models/words_labels.txt")

# ---- Labels ----
labels1 = ["1","2","3","4","5","6","7","8","9"]
labels2 = ["A","B","C","D","E","F","G","H","I","J",
           "K","L","M","N","O","P","Q","R","S","T",
           "U","V","W","X","Y","Z"]
labels3 = ["Afraid","Agree","Bad","Become","Doctor",
           "Pain","Pray","Secondary","Skin","Small",
           "Specific","Today","Warn","Work","You","Stand"]

# ---- Parameters ----
offset = 20
imgSize = 300
current_model = 1       # start with numbers model
confidence_threshold = 0.85

# ---- Keep track of last prediction to avoid duplicates ----
last_label = None

while True:
    success, img = cap.read()
    if not success:
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    # ---- Switch models ----
    key = cv2.waitKey(1)
    if key == ord("1"):
        current_model = 1
        print("Switched to Numbers model")
    elif key == ord("2"):
        current_model = 2
        print("Switched to Alphabets model")
    elif key == ord("3"):
        current_model = 3
        print("Switched to Words model")

    if hands:
        # ---- Combine hands if two ----
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']
            x_min = min(x1, x2) - offset
            y_min = min(y1, y2) - offset
            x_max = max(x1 + w1, x2 + w2) + offset
            y_max = max(y1 + h1, y2 + h2) + offset
        else:
            x, y, w, h = hands[0]['bbox']
            x_min, y_min, x_max, y_max = x - offset, y - offset, x + w + offset, y + h + offset

        # Clamp to frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img.shape[1], x_max), min(img.shape[0], y_max)

        imgCrop = img[y_min:y_max, x_min:x_max]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if imgCrop.size != 0:
            h, w, _ = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # ---- Run the selected model ----
            if current_model == 1:
                prediction, index = classifier1.getPrediction(imgWhite, draw=False)
                if prediction[index] > confidence_threshold and index < len(labels1):
                    label = labels1[index]
                    conf = prediction[index]
                    cv2.putText(imgOutput, f"Number: {label} ({conf*100:.1f}%)",
                                (x_min, y_min - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            elif current_model == 2:
                prediction, index = classifier2.getPrediction(imgWhite, draw=False)
                if prediction[index] > confidence_threshold and index < len(labels2):
                    label = labels2[index]
                    conf = prediction[index]
                    cv2.putText(imgOutput, f"Alphabet: {label} ({conf*100:.1f}%)",
                                (x_min, y_min - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            elif current_model == 3:
                prediction, index = classifier3.getPrediction(imgWhite, draw=False)
                if prediction[index] > confidence_threshold and index < len(labels3):
                    label = labels3[index]
                    conf = prediction[index]
                    cv2.putText(imgOutput, f"Word: {label} ({conf*100:.1f}%)",
                                (x_min, y_min - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # ---- Insert into DB only if label changed ----
            if prediction[index] > confidence_threshold and label != last_label:
                insert_prediction(label, conf)
                last_label = label

            # Draw rectangle & previews
            cv2.rectangle(imgOutput, (x_min, y_min), (x_max, y_max), (255, 0, 255), 4)
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)