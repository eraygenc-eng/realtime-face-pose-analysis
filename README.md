## Real-Time Face and Pose Analysis with OpenCV & MediaPipe

This project is a real-time computer vision application developed with **Python**, **OpenCV**, and **MediaPipe**.

The application uses a webcam to detect the user's face and body landmarks in real time. Based on these landmarks, it performs simple but useful movement and face analyses such as eye open/closed detection, hand raise detection, arm angle analysis, jump counting, and basic posture analysis.

The project is designed not only as a computer vision learning project, but also as a fun interactive webcam application. Users can move in front of the camera and instantly see feedback on the screen based on their body position, gestures, and facial state.

---

# Overview

The main idea of this project is to process webcam frames in real time and analyze human movement using landmark detection.

Instead of using only one long Python file, the project is organized into separate modules. This makes the code easier to understand, debug, improve, and extend.

The application currently includes two main analysis systems:

1. **Face Analysis**
2. **Pose Analysis**

The face analysis module focuses on detecting the face mesh and analyzing eye state.

The pose analysis module focuses on detecting body landmarks and analyzing body movements such as raised hands, arm position, jumping, and posture.

---

# Features

## Face Analysis

The face analysis part uses **MediaPipe Face Mesh** to detect detailed facial landmarks.

Current face-related features include:

- Real-time face mesh detection
- Face landmark visualization
- Eye open / closed detection
- Eye distance calculation
- Real-time text feedback on the screen

The eye detection logic is based on the distance between selected upper and lower eye landmarks. If the calculated average eye distance goes below a certain threshold, the system classifies the eyes as closed. Otherwise, it classifies them as open.

This can be used as a basic example for understanding attention tracking, blink detection, and face landmark-based analysis.

---

## Pose Analysis

The pose analysis part uses **MediaPipe Pose** to detect body landmarks.

Current pose-related features include:

- Real-time pose landmark detection
- Body skeleton visualization
- Left hand raise detection
- Right hand raise detection
- Both hands up detection
- Left arm straight / bent detection
- Right arm straight / bent detection
- Jump counter
- Basic posture analysis
- Real-time movement feedback

The arm analysis is based on calculating the angle between shoulder, elbow, and wrist landmarks. If the angle is high, the arm is considered straight. If the angle is lower, the arm is considered bent.

The jump counter uses the vertical movement of the hip landmarks. When the hip position moves upward beyond a defined threshold and then returns close to the baseline position, the system counts it as a jump.

The posture analysis uses shoulder alignment and the relative position of the head and shoulders to give a simple posture status.

---

# Technologies Used

This project uses the following technologies:

- **Python**  
  Main programming language used in the project.

- **OpenCV**  
  Used for webcam access, frame processing, image display, text drawing, and real-time video handling.

- **MediaPipe**  
  Used for face mesh and pose landmark detection.

---

# Using a Phone as a Webcam

If you do not have a built-in webcam or an external webcam on your computer, you can use your phone as a webcam.

One option is to install **Iriun Webcam** on both your computer and your phone. After connecting your phone and computer to the same network, Iriun Webcam can make your phone camera available as a webcam input.

After setting it up, you can run the project normally.

---


## How to Run This Project

Follow these steps to run the project on your computer.

# 1. Clone the repository

```bash
git clone https://github.com/eraygenc-eng/realtime-face-pose-analysis.git
```

# 2. Go into the project folder

```bash
cd realtime-face-pose-analysis
```

# 3. Install the required packages

```bash
pip install -r requirements.txt
```

# 4. Run the application

```bash
python main.py
```

# 5. Quit the application

When the webcam window is open, press:

```text
q
```

to close the application.


---

# WARNING !

If the wrong camera opens, you may need to change the camera index in `config.py` and try changing it to "CAMERA_INDEX=1" or "CAMERA_INDEX=2"

depending on which camera input your system detects.

---


# Project Structure

- `main.py` - Main entry point of the application
- `config.py` - Configuration values and thresholds
- `requirements.txt` - Required Python packages
- `.gitignore` - Files and folders ignored by Git
- `README.md` - Project documentation

## Folders

- `analyzers/`
  - `face_analyzer.py` - Face mesh and eye open/closed analysis
  - `pose_analyzer.py` - Pose, gesture, jump, and posture analysis

- `display/`
  - `draw_text.py` - Helper function for drawing text on frames

- `utils/`
  - `geo.py` - Geometry helper functions such as angle and distance calculation
