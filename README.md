# How is the project structured
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
## What if my IP address of the Flask server is not http://127.0.0.1:5000
In the textfield on your localhost URL (the Vite server), please put in the URL of the different Flask server (do not put a / at the end of the URL). For example, only put `http://127.0.0.1:5000` not `http://127.0.0.1:5000/`.

# Why so complicated?
We deployed all of this already, however, it wasn't optimal. Firstly, Flask is a full-stack web framework, hence, the video from the camera would have to be sent to the server. If deployed on the cloud, the latency to send the video feed to the server and back is too laggy and unusable. We tried this and it was not a good UX. So, we thought we could just deploy the Vite server and not the Flask server. However, if the Vite is a HTTPS server, the embedded `<iframe>` HTML page has to also be HTTPS. It is a browser policy. After all that, we decided that self-hosting it on your Mac is the best possible option.

We apologise for the inconvenience. If it doesn't work and it doesn't go against PT marking "fairness", we are happy to debug the issue WITHOUT changing the code should there be dependency issues.
