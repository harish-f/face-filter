# How the project is structured
The project is structured across 2 repos, this is the repo for the Flask server, and this [repo](https://github.com/SeansC12/face_filter_web) is the Vite server. Why is it like this? The Vite server accesses the Flask camera feed via the HTML element `<iframe>`

# How to run
## 1. Clone both repos
```bash 
git clone https://github.com/harish-f/face-filter
```
and
```bash 
git clone https://github.com/SeansC12/face_filter_web
```
## 2. Run the Flask Server
You may need to install additional dependencies like `Flask` and `mediapipe`. If uninstalled, just install them with pip.

cd into your `face-filter` directory. Then, run this command
```bash
flask run --debug
```
After that, it will tell you which IP address it is being served at. Ensure that this IP address is http://127.0.0.1:5000, if it not, change it in the textfield which will be displayed later.

<img width="567" alt="Flask_server" src="https://github.com/harish-f/face-filter/assets/88256324/fdf715d3-6a50-4ba1-a53e-bf54ac4d75a7">

## 3. Run the Vite Server
[Install Node.js](https://treehouse.github.io/installation-guides/mac/node-mac.html) on your Mac if you haven't already.

cd into your `face_filter_web` directory. Then, run this command
```bash
npm i
```
Then, run this,
```bash
npm run dev
```
After this, go to the localhost URL given by the CLI tool, then enjoy.

<img width="569" alt="Vite_server" src="https://github.com/harish-f/face-filter/assets/88256324/0c98156d-40a7-460e-906f-9fc5692f3ae3">

## What if my IP address of the Flask server is not http://127.0.0.1:5000
In the textfield on your localhost URL (the Vite server), please put in the URL of the different Flask server (do not put a / at the end of the URL). For example, only put `http://127.0.0.1:5000` not `http://127.0.0.1:5000/`.

# How to use the app
There are 3 main sections of the app
- The theme/textbox for Flask server address
- Video feed (`iframe`)
- Carousell for selection of prop

## Carousel colour scheme to select props
<img width="519" alt="image" src="https://github.com/harish-f/face-filter/assets/88256324/d05a63f1-09af-4809-b946-ebf464d6687f">

- The colour green represents what prop is currently enabled.
- The indigo gradient represents the uniqueness of the moustache prop, due to its ability to be adjusted based on your finger pinch height

# Why unnecessarily complicated to host?
Our initial plan was to deploy all of this on the cloud, so that you need not self-host. In fact, we deployed all of this already, however, it wasn't optimal. Firstly, Flask is a full-stack web framework, hence, the video from the camera would have to be sent to the server (location of hosting). If deployed on the cloud, the latency to send the video feed to the server for processing and back is too laggy and unusable. We tried this and it was not a good UX. So, we thought we could just deploy the Vite server and not the Flask server. However, if the Vite is a HTTPS server, the embedded `<iframe>` HTML page also has to be HTTPS. It is a browser policy. After all that, we decided that self-hosting both servers on your Mac is the best possible option.

We apologise for the inconvenience. If it doesn't work and it doesn't go against PT marking "fairness", we are happy to debug the issue WITHOUT changing the code should there be dependency issues.

# Demonstration
We uploaded a demonstration on Google drive [here](https://drive.google.com/file/d/1jIw62yph2-JvK4hNi9jO19rJt8XAwiPl/view?usp=sharing).
