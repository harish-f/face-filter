# from imutils import face_utils
# import dlib
# import cv2

# p = "shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(p)

# cap = cv2.VideoCapture(0)





# import mediapipe as mp
# import cv2

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# capture = cv2.VideoCapture(0)

# with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
  
#     while capture.isOpened():
        
#         _, image = cap.read()

#         # Converting the image to gray scale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#         # Get faces into webcam's image
#         rects = detector(gray, 0)

#         # For each detected face, find the landmark.
#         for (i, rect) in enumerate(rects):
#             # Make the prediction and transfom it to numpy array
#             shape = predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)

#             # Draw on our image, all the finded cordinate points (x,y) 
#             for (x, y) in shape:
#                 cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

#         ret, frame = capture.read()
#         frame = cv2.flip(frame, 1)
#         # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detected_image = hands.process(image)
#         # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if detected_image.multi_hand_landmarks:
#             for hand_lms in detected_image.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, hand_lms,
#                                         mp_hands.HAND_CONNECTIONS,
#                                         landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                                             color=(255, 0, 255), thickness=4, circle_radius=2),
#                                         connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
#                                             color=(20, 180, 90), thickness=2, circle_radius=2)
#                                         )

#         cv2.imshow('Webcam', image)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# capture.release()
# cv2.destroyAllWindows()

from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
