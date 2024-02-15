# def overlay_image(frame, overlay_path, keypoint):
#     # Load the overlay image
#     image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

#     desired_width, desired_height = resize_overlay_image(True, image, 50)

#     # Resize the image
#     overlay_image = cv2.resize(image, (desired_width, desired_height))

#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     # Check if at least one face is detected
#     if len(faces) == 0:
#         return frame

#     # Iterate through detected faces
#     for face in faces:
#         # Detect facial keypoints
#         landmarks = predictor(gray, face)

#         # Convert dlib landmarks to numpy array
#         landmarks_points = np.array([[landmark.x, landmark.y] for landmark in landmarks.parts()])

#         # Extract the specified keypoint coordinates
#         x, y = landmarks_points[keypoint]

#         # Get the size of the overlay image
#         overlay_height, overlay_width, _ = overlay_image.shape

#         # Calculate the position to overlay the image
#         x1 = int(x - overlay_width / 2)
#         y1 = int(y - overlay_height / 2)
#         x2 = int(x1 + overlay_width)
#         y2 = int(y1 + overlay_height)

#         # Ensure the overlay image fits within the frame boundaries
#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(frame.shape[1], x2)
#         y2 = min(frame.shape[0], y2)

#         # Resize the overlay image to fit the region of interest
#         overlay_image_resized = cv2.resize(overlay_image, (x2 - x1, y2 - y1))

#         # Overlay the resized image onto the frame
#         alpha_s = overlay_image_resized[:, :, 3] / 255.0
#         alpha_l = 1.0 - alpha_s
#         for c in range(0, 3):
#             frame[y1:y2, x1:x2, c] = (alpha_s * overlay_image_resized[:, :, c] +
#                                        alpha_l * frame[y1:y2, x1:x2, c])

#     return frame

from flask import Flask, render_template, Response, request
import cv2
import dlib
import numpy as np

app = Flask(__name__)

port = 8080
prop_selected = 0

# Load the pre-trained facial landmark detector model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def resize_overlay_image(keepDimension, image, width, height=0):
    if keepDimension and height == 0:
        desired_width = width

        # Calculate the aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]

        # Calculate the corresponding height based on the aspect ratio
        desired_height = int(desired_width / aspect_ratio)

        return desired_width, desired_height
    
    return width, height

# Function to overlay image over specified facial keypoint
def overlay_image(frame, overlay_path, prop_index):
    # Load the overlay image
    image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    
    if prop_index == 0:
        desired_width, desired_height = resize_overlay_image(False, image, 150, 50)
    elif prop_index == 1:
        desired_width, desired_height = resize_overlay_image(True, image, 380)
    elif prop_index == 2:
        desired_width, desired_height = resize_overlay_image(True, image, 380)

    # Resize the image
    overlay_image = cv2.resize(image, (desired_width, desired_height))

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Check if at least one face is detected
    if len(faces) == 0:
        return frame

    # Iterate through detected faces
    for face in faces:
        # Detect facial keypoints
        landmarks = predictor(gray, face)

        # Convert dlib landmarks to numpy array
        landmarks_points = np.array([[landmark.x, landmark.y] for landmark in landmarks.parts()])

        # Extract the specified keypoint coordinates
        if prop_index == 0:
            # These 2 landmarks are the bottom of the nose and top of the lip,
            # getting the middle will give the best position for a moustache
            a, b = landmarks_points[34]
            c, d = landmarks_points[52]
            
            x = (a + c) / 2
            y = (b + d) / 2
        elif prop_index == 1:
            a, b = landmarks_points[22]
            c, d = landmarks_points[23]

            x = (a + c) / 2
            y = ((b + d) / 2) - 100
        elif prop_index == 2:
            a, b = landmarks_points[22]
            c, d = landmarks_points[23]

            x = (a + c) / 2
            y = ((b + d) / 2) - 100

        # Get the size of the overlay image
        overlay_height, overlay_width, _ = overlay_image.shape

        # Calculate the position to overlay the image
        x1 = int(x - overlay_width / 2)
        y1 = int(y - overlay_height / 2)
        x2 = int(x1 + overlay_width)
        y2 = int(y1 + overlay_height)

        # Ensure the overlay image fits within the frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Resize the overlay image to fit the region of interest
        overlay_image_resized = cv2.resize(overlay_image, (x2 - x1, y2 - y1))

        # Overlay the resized image onto the frame
        alpha_s = overlay_image_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * overlay_image_resized[:, :, c] +
                                       alpha_l * frame[y1:y2, x1:x2, c])

    return frame


def video_feed():
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(1)

    # Load the overlay image
    overlay_path = ["assets/moustache.png", "assets/propellor_hat.png", "assets/cap_hat.png"]

    while True:
        global prop_selected
        ret, frame = cap.read()

        if not ret:
            break

        # Overlay the image with the specified keypoint
        # print(prop_selected, "hi")
        frame = overlay_image(frame, overlay_path[prop_selected], prop_selected)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert the frame to bytes
        frame_bytes = buffer.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the camera and close the window
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/switch_prop")
def switch_prop():
    global prop_selected
    print(int(request.args["prop"]))
    prop_selected = int(request.args["prop"])
    return

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
