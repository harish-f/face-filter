from imutils import face_utils
import dlib
from flask import Flask, render_template, Response
import cv2
import numpy as np

# filters = [
#     {name: "thing", file: "stache.png"},
# ]


app = Flask(__name__)


p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Initialize the camera
camera = cv2.VideoCapture(0)

overlay_path = 'overlay_image.png'
overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

def overlay_nose(frame, nose_position, faceWidth):
    resized_overlay = cv2.resize(overlay,(faceWidth, faceWidth))
    
    # Get the dimensions of the overlay image
    overlay_height, overlay_width, _ = resized_overlay.shape

    # Calculate the position to overlay the image onto the nose
    x_offset = nose_position[0] - overlay_width // 2
    y_offset = nose_position[1] - overlay_height // 2

    # Ensure the overlay stays within the frame boundaries
    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + overlay_width)
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + overlay_height)

    alpha_s = resized_overlay[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for y in range(y1, y2):
        for x in range(x1, x2):
            frame[y,x,0] = resized_overlay[y-y1,x-x1,0]
            frame[y,x,1] = resized_overlay[y-y1,x-x1,1]
            frame[y,x,2] = resized_overlay[y-y1,x-x1,2]
    return frame


def generate_frames():
    while True:
        # Getting out image by webcam 
        success, image = camera.read()

        if not success:
            break
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                if len(shape) < 30: continue
                image = overlay_nose(image, (shape[30][0], shape[30][1]), shape[16][0] - shape[0][0])
            # except: None

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/input")
def getInput():
    return "ok"

if __name__ == '__main__':
    app.run(debug=True)
