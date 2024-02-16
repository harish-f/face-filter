from flask import Flask, render_template, Response, request
import cv2
import dlib
import numpy as np
import mediapipe as mp

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load the pre-trained facial landmark detector model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


prop_selected = 0

def getFaceWidth(keypoints):
    return keypoints[16][0] - keypoints[0][0]

def getFaceHeight(keypoints):
    return keypoints[8][1] - keypoints[19][1]

def isClicked(keypoints): # using pythagoras theorom to determine the distance between thumb and index fingertip
    thumb_tip = keypoints[4]
    index_tip = keypoints[8]
    dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5 # returned as a float
    return dist < 0.04

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
def overlay_image(frame, overlay_path, prop_index, scaleVal):
    # Load the overlay image
    image = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

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

        # resize selected prop based on face width and preset constant
        if prop_index == 0:
            desired_width, desired_height = resize_overlay_image(True, image, int(getFaceWidth(landmarks_points)*scaleVal))
        elif prop_index == 1:
            desired_width, desired_height = resize_overlay_image(True, image, int(getFaceWidth(landmarks_points)*1.3*scaleVal))
        elif prop_index == 2:
            desired_width, desired_height = resize_overlay_image(True, image, int(getFaceWidth(landmarks_points)*2*scaleVal))
        elif prop_index == 3:
            desired_width, desired_height = resize_overlay_image(True, image, int(getFaceWidth(landmarks_points)*1*scaleVal))

        # Resize the image
        overlay_image = cv2.resize(image, (desired_width, desired_height))

        # Extract the specified keypoint coordinates
        if prop_index == 0:
            # These 2 landmarks are the bottom of the nose and top of the lip,
            # getting the middle will give the best position for a moustache
            a, b = landmarks_points[33]
            c, d = landmarks_points[51]
            
            x = (a + c) / 2
            y = (b + d) / 2
        elif prop_index == 1:
            a, b = landmarks_points[21]
            c, d = landmarks_points[22]

            y_offset = (landmarks_points[8][1] - landmarks_points[51][1]) * 1.5 # offset of hat based on distance from chin to upper lip

            x = (a + c) / 2
            y = landmarks_points[19][1] - y_offset
        elif prop_index == 2:
            a, b = landmarks_points[21]
            c, d = landmarks_points[22]

            y_offset = landmarks_points[8][1] - landmarks_points[51][1] # offset of hat based on distance from chin to upper lip
            
            x = (a + c) / 2
            y = landmarks_points[19][1] - y_offset*1.3
        elif prop_index == 3:
            a, b = landmarks_points[28]
            c, d = landmarks_points[28]
            
            x = (a + c) / 2
            y = (b + d) / 2
        elif prop_index == 4:
            a, b = landmarks_points[28]
            c, d = landmarks_points[28]
            
            x = (a + c) / 2
            y = (b + d) / 2
        elif prop_index == 5:
            a, b = landmarks_points[28]
            c, d = landmarks_points[28]
            
            x = (a + c) / 2
            y = (b + d) / 2

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
    cap = cv2.VideoCapture(0)
    
    
    # Reading resolution of camera input
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    frameWidth = len(frame[0])
    frameHeight = len(frame)
    
    
    clicked = False # initialise variable to check if finger is clicked
    initialFinger = 0 # initalise start position of finger gesture
    sliderVal = 100 # initialise slider value 0 to 255
    sliderSize = 30 # specify width of slider

    sliderBase = np.array([[255]*10]*(frameHeight-20)) # create slider base to be inserted into image
    sliderButton = np.array([[0]*25]*sliderSize) # create slider shape to be inserted into image

    # Load the overlay image
    overlay_path = ["assets/moustache.png", "assets/propellor_hat.png", "assets/cap_hat.png", "assets/glasses.png", "assets/cool_glasses.png", "assets/graduation_glasses_transparent.png"]

    while True:
        global prop_selected
        ret, frame = cap.read()

        if not ret:
            break

        # calculating the slider's position relative to the image
        if sliderVal >= 255:
            sliderVal = 254
        elif sliderVal == 0:
            sliderVal = 0
        buttonPos = int(sliderVal/(sliderSize + 255)*frameHeight)

        # inserting slider module into the image
        if prop_selected == 0:
            for c in range(3):
                frame[10:-10,-30:-20,c] = sliderBase
                frame[buttonPos:buttonPos+sliderSize,-37:-12,c] = sliderButton


        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            
            # processing the image to get the hand keypoints
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_image = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if detected_image.multi_hand_landmarks:
                for hand_keypoints in detected_image.multi_hand_landmarks: # Check all hands
                    keypoints = hand_keypoints.landmark
                    if clicked:
                        if not isClicked(keypoints):
                            # reset clicked and initial finger vars if stopped clicking
                            clicked = False
                            initialFinger = 0

                            continue

                        # calculating change in sliderVal based on y movement of finger
                        sliderVal += (keypoints[4].y - initialFinger) * (255 + sliderSize) * 2
                        initialFinger = keypoints[4].y

                    elif isClicked(keypoints):
                        # changes clicked var to true if click is detected and clicked was initially false
                        clicked = True
                        initialFinger = keypoints[4].y
        
        # Overlay the image with the specified keypoint
        frame = overlay_image(frame, overlay_path[prop_selected], prop_selected, (sliderVal/255 + 0.5) if prop_selected == 0 else 1)
            

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
    app.run(debug=True, port=5050)
