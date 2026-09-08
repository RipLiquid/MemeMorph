# MemeMorph

MemeMorph is a real-time computer vision application that detects facial expressions from a webcam and dynamically triggers animated reaction overlays.

The project uses MediaPipe facial landmarks and facial blendshapes to analyze expressions such as eye closure, squinting, eyebrow movement, and lip shape. OpenCV processes the live webcam stream while animated reaction GIFs are positioned over the detected face.

## Current Demo

MemeMorph currently supports its first reaction:

**Reverse Smile Reaction**

The reaction is detected using a combination of:

- Eyes closed **or** squinted
- Eyebrows lowered
- Matching lip movement
- Temporal smoothing to reduce accidental triggers

When the expression is detected, MemeMorph plays an animated reaction GIF over the user's face and tracks the face as it moves.

## Features

- Real-time webcam processing
- 1280x720 16:9 camera output
- MediaPipe Face Landmarker
- Facial landmark tracking
- Facial blendshape analysis
- Eye closure detection
- Eye squint detection
- Eyebrow movement detection
- Lip-shape analysis
- Multi-feature expression matching
- Animated GIF reaction overlays
- Face-relative overlay positioning
- Reaction hold-time filtering
- Release grace period to reduce flickering
- Live debugging interface
- Manual reaction testing

## Controls

| Key | Action |
|---|---|
| `Q` | Quit MemeMorph |
| `D` | Toggle debugging information |
| `G` | Manually test the current reaction GIF |

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pillow

## Architecture

```text
Webcam
   |
   v
OpenCV
   |
   v
MediaPipe Face Landmarker
   |
   +----------------------+
   |                      |
   v                      v
478 Face Landmarks   Facial Blendshapes
                          |
                          v
                  Expression Analysis
                          |
                          v
                   Reaction Engine
                          |
                          v
                  Animated GIF Overlay
                          |
                          v
                    Final Webcam
```

## Expression Detection

MemeMorph does not rely on a single facial measurement.

The first reaction combines several facial features:

```text
Eyes CLOSED
      OR
Eyes SQUINTED
       +
Eyebrows DOWN
       +
Matching lip movement
       |
       v
Reaction Trigger
```

Using multiple facial signals reduces false detections caused by normal blinking or unrelated facial movements.

## Project Structure

```text
MemeMorph/
|
|-- assets/
|   `-- reactions/
|
|-- models/
|   `-- face_landmarker.task
|
|-- src/
|   `-- main.py
|
|-- tests/
|
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/RipLiquid/MemeMorph.git
cd MemeMorph
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Download the MediaPipe model

Download the MediaPipe Face Landmarker model and place it at:

```text
models/face_landmarker.task
```

The application expects the model at this exact location.

### 5. Add a reaction asset

Reaction media is not included in the repository.

Add your own GIF as:

```text
assets/reactions/speed_reverse_smile.gif
```

Third-party reaction media is excluded from Git so MemeMorph's source code can remain separate from externally owned media.

### 6. Run MemeMorph

```powershell
python src/main.py
```

## Debug Mode

MemeMorph includes a live debug panel showing facial-expression measurements such as:

```text
eyeBlink
eyeSquint
eyeWide
browDown
browInnerUp
jawOpen
mouthPucker
mouthFunnel
mouthPress
mouthRoll
mouthStretch
```

The panel also displays whether the eyes, brows, and lips currently satisfy the reaction rules.

Press `D` to hide the debugging interface.

## Roadmap

### Computer Vision

- [x] Webcam processing
- [x] Facial landmark tracking
- [x] Facial blendshape analysis
- [x] First facial reaction
- [x] Animated GIF overlays
- [ ] Multiple reaction expressions
- [ ] Reaction configuration system
- [ ] Face rotation-aware overlays
- [ ] Temporal expression confidence system

### Body and Gesture Recognition

- [ ] MediaPipe Pose Landmarker
- [ ] Shoulder tracking
- [ ] Elbow tracking
- [ ] Wrist tracking
- [ ] MediaPipe Hand Landmarker
- [ ] Finger tracking
- [ ] Hand gesture recognition
- [ ] Combined face + body reaction detection

Future reactions will be able to use combinations such as:

```text
Facial Expression
       +
Arm Position
       +
Hand Gesture
       |
       v
Reaction
```

### Cloud

A future version of MemeMorph is planned to integrate Microsoft Azure while maintaining a zero-cost development architecture.

Potential Azure functionality includes:

- Azure App Configuration
- Remotely configurable reaction thresholds
- Reaction feature flags
- Configurable detection parameters
- Local fallback configuration
- Azure Static Web Apps project/demo site

Computer vision processing will remain local so webcam footage does not need to be uploaded to the cloud.

### Virtual Camera

Future versions may output the processed video as a virtual webcam for use with applications such as:

- Discord
- Microsoft Teams
- Zoom
- OBS

## Privacy

MemeMorph performs computer vision processing locally.

Webcam frames are processed on the user's computer and are not currently uploaded to external services.

## Status

**MemeMorph v0.1 — Initial Computer Vision Prototype**

Current focus:

> Real-time facial-expression detection with animated reaction overlays.
