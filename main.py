import cv2
from simple_facerec import SimpleFacerec
from tensorflow.keras.models import model_from_json
import numpy as np

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("C:\\Users\\Anandini\\Downloads\\images")

# Load Camera
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("Face_Antispoofing_System\\models\\haarcascade_frontalface_default.xml")
json_file = open("Face_Antispoofing_System\\antispoofing_models\\antispoofing_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("Face_Antispoofing_System\\antispoofing_models\\antispoofing_model.h5")
print("Model loaded from disk")

video = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = video.read()

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            # Select the face with the largest area
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

            # Scale the face region back to the original size
            x, y, w, h = x * 2, y * 2, w * 2, h * 2

            face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)

            preds = model.predict(resized_face)[0]
            print(preds)
            if preds > 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
            else:
                # Detect Faces
                face_locations, face_names = sfr.detect_known_faces(frame)

                if len(face_locations) > 0:
                    # Calculate distance from the camera for each detected face
                    distances = [sum(face_loc) for face_loc in face_locations]

                    # Find the index of the closest face
                    closest_index = distances.index(min(distances))

                    # Retrieve the closest face location and name
                    closest_face_loc = face_locations[closest_index]
                    closest_face_name = face_names[closest_index]

                    # Draw bounding box and display name for the closest face
                    y1, x2, y2, x1 = closest_face_loc
                    cv2.putText(frame, closest_face_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
    
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()



