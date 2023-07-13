import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from scipy.spatial import distance as dist

root_dir = os.getcwd()

face_cascade = cv2.CascadeClassifier("Face_Antispoofing_System\\models\\haarcascade_frontalface_default.xml")

json_file = open("Face_Antispoofing_System\\antispoofing_models\\antispoofing_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("Face_Antispoofing_System\\antispoofing_models\\antispoofing_model.h5")
print("Model loaded from disk")

video = cv2.VideoCapture(0)

# Load registered face embeddings and labels from "images" folder
images_folder = "C:\\Users\\Anandini\\Downloads\\images"
registered_faces = {}

for image_name in os.listdir(images_folder):
    image_path = os.path.join(images_folder, image_name)
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (160, 160))
        resized_image = resized_image.astype("float") / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)

        image_label = image_name.split(".")[0]
        registered_faces[image_label] = model.predict(resized_image)[0]

while True:
    try:
        ret, frame = video.read()

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minSize=(30, 30))
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
                matched_label = None
                min_distance = float('inf')

                for label, registered_face in registered_faces.items():
                    d = dist.euclidean(resized_face.flatten(), registered_face.flatten())

                    if d < min_distance:
                        min_distance = d
                        matched_label = label

                if matched_label and min_distance < 0.5:  # Adjust the threshold here
                    # Print the matched label (name) above the face
                    cv2.putText(frame, matched_label, (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'unregistered', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass

video.release()
cv2.destroyAllWindows()
