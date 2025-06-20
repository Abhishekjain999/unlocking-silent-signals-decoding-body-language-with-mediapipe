from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                pose = results.pose_landmarks.landmark
                face = results.face_landmarks.landmark
                pose_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in pose]).flatten())
                face_row = list(np.array([[l.x, l.y, l.z, l.visibility] for l in face]).flatten())
                row = pose_row + face_row
                X = pd.DataFrame([row])
                predicted_class = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                cv2.putText(image, f'CLASS: {predicted_class}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image, f'PROB: {np.max(prob):.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            except:
                pass

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
