# MemeMorph
![Python](https://img.shields.io/badge/Python-3.11+-blue) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-Real--Time-green) ![Azure](https://img.shields.io/badge/Azure-App%20Configuration-0078D4) ![CI](https://github.com/RipLiquid/MemeMorph/actions/workflows/ci.yml/badge.svg)

**MemeMorph** is a real-time computer vision reaction engine that detects facial expressions, hand gestures, and body poses from a webcam and triggers animated meme overlays. It combines MediaPipe face, hand, and pose tracking with personalized neutral-face calibration, configurable reaction thresholds, automated tests, GitHub Actions CI, and optional Azure App Configuration.

## Features
- Real-time facial expression, hand gesture, and body pose tracking
- 10 reaction types with animated overlays
- Personalized neutral-face calibration to reduce false triggers
- 1920×1080 output with lower-resolution AI inference for better performance
- Independent face, hand, and pose detection cadences
- Reaction priority system for overlapping gestures
- Local JSON configuration with optional Azure App Configuration overrides
- 30 automated unit tests covering reactions, calibration, geometry, and priority
- GitHub Actions CI on pushes and pull requests
- Automated Windows setup script
- Manual reaction testing and debug controls

## Reactions
| Key | Reaction | Detection |
|---|---|---|
| `1` | Speed Reverse Smile | Eye squint/blink change + brow-down change + closed jaw |
| `2` | Eyebrow Raise | Strong single-eyebrow asymmetry |
| `3` | Surprised Pikachu | Wide eyes + open jaw |
| `4` | Sad / Cry | Inner brow raise + frown |
| `5` | Side Eye | Strong sideways eye movement |
| `6` | Jerry Laugh | Smile + open jaw + squint/blink |
| `7` | Jerry Point & Laugh | Sideways thumb point + laughing/smirking face |
| `8` | Facepalm | Open palm positioned over the upper face |
| `9` | Thumbs Up | Raised thumb with folded fingers |
| `0` | Absolute Cinema | Two open hands raised and spread wider than the shoulders |

## How It Works
```text
Webcam → 16:9 frame → MediaPipe Face / Hands / Pose → Neutral calibration → Feature metrics → Reaction engine → Priority + smoothing → Animated overlay → 1080p output
```

MemeMorph does not rely on one measurement. Reactions are built from combinations of facial and gesture signals. For example, the Speed reaction uses changes from the user's calibrated neutral face:
```text
Eyes squinted OR blink change
            +
      Eyebrows down
            +
        Jaw closed
            ↓
      Reaction trigger
```
Using multiple signals and a personalized baseline helps reduce false detections caused by normal blinking, resting expressions, and unrelated movement.

## Personalized Calibration
At startup, MemeMorph learns the user's neutral expression before automatic reactions are enabled:
```text
WARMING UP → LEARNING NEUTRAL → VALIDATING → ARMING REACTION ENGINE → READY
```
The neutral baseline is built from median samples instead of a single frame, making it more resistant to noisy readings and temporary tracking errors.

## Performance
The interface stays at **1920×1080**, while AI inference runs on a smaller **640×360** frame. MediaPipe landmarks are normalized, so detections still map correctly to the full-resolution output.
```text
Face detection: every frame
Hand detection: every 2 frames
Pose detection: every 4 frames
Face landmark drawing: reduced when enabled
```
This keeps the UI sharp while reducing unnecessary inference work.

## Reaction Priority
Some gestures can overlap, so the reaction engine evaluates more specific gestures first:
```text
Absolute Cinema
→ Facepalm
→ Jerry Point
→ Thumbs Up
→ Speed
→ Eyebrow Raise
→ Surprised
→ Sad
→ Side Eye
→ Jerry Laugh
```
The priority list and detector registry are isolated in `src/reaction_engine.py`, making future reactions easier to add without modifying the main application loop.

## Azure App Configuration
MemeMorph works completely locally by default. `config/reaction_defaults.json` provides the fallback settings, while Azure App Configuration can optionally override thresholds at startup.
```text
Local JSON defaults
        ↓
Azure available? ── No ──→ Use local settings
        │
       Yes
        ↓
Apply MemeMorph:* overrides
        ↓
Final runtime configuration
```
Example Azure keys:
```text
MemeMorph:reactions:speed:squint_delta_min
MemeMorph:reactions:speed:brow_down_delta_min
MemeMorph:reactions:eyebrow:brow_asymmetry_delta_min
```
Set the endpoint for the current PowerShell session:
```powershell
$Env:AZURE_APPCONFIG_ENDPOINT="https://YOUR-STORE.azconfig.io"
```
Authentication uses `DefaultAzureCredential`, so Azure credentials are not hard-coded into the repository. Webcam frames remain local; Azure is used only for optional configuration values.

## Controls
| Key | Action |
|---|---|
| `D` | Toggle debug panel |
| `L` | Toggle landmarks |
| `R` | Recalibrate neutral face |
| `Q` | Quit |
| `1`–`9`, `0` | Manually trigger reactions for testing |

## Quick Start
### Windows
```powershell
git clone https://github.com/RipLiquid/MemeMorph.git
cd MemeMorph
.\scripts\setup.ps1
.\.venv\Scripts\python.exe src\main.py
```
The setup script:
- checks for Python
- creates `.venv`
- installs dependencies
- creates required directories
- downloads the MediaPipe face, hand, and pose model files
- checks reaction media
- detects Azure configuration
- runs the automated test suite

Third-party meme GIFs are intentionally not committed to the public repository. Add your own media to `assets/reactions/` using the expected filenames shown by the setup script.

## Manual Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\main.py
```

## Testing
Run the complete test suite without a webcam:
```powershell
python -m unittest discover -s tests -v
```
Current coverage includes:
- neutral baseline and delta calculations
- Speed and eyebrow calibration logic
- surprised, sad, side-eye, and Jerry reactions
- thumbs-up and thumb-point geometry
- facepalm and Absolute Cinema detection
- reaction enable/disable configuration
- detector registry consistency
- reaction priority behavior

Current suite:
```text
30 tests
OK
```

## Continuous Integration
GitHub Actions runs the test suite automatically on every push and pull request to `main`.
```text
Checkout → Python 3.11 → Install dependencies → Run unit tests → Pass / Fail
```
Workflow: `.github/workflows/ci.yml`

## Project Structure
```text
MemeMorph/
├── .github/
│   └── workflows/
│       └── ci.yml
├── assets/
│   └── reactions/                 # Local reaction media; ignored by Git
├── config/
│   └── reaction_defaults.json     # Local runtime defaults
├── models/                        # MediaPipe .task models; ignored by Git
├── scripts/
│   └── setup.ps1                  # Automated Windows setup
├── src/
│   ├── app_config.py              # Local + Azure configuration loader
│   ├── hand_tracker.py            # MediaPipe hand tracking
│   ├── main.py                    # Camera, calibration, UI, rendering loop
│   ├── pose_tracker.py            # MediaPipe pose tracking
│   └── reaction_engine.py         # Detectors, registry, priority, thresholds
├── tests/
│   └── test_reactions.py          # 30 automated unit tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Technology Stack
| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Computer Vision | OpenCV |
| Face Tracking | MediaPipe Face Landmarker |
| Hand Tracking | MediaPipe Hand Landmarker |
| Pose Tracking | MediaPipe Pose Landmarker |
| Image / GIF Processing | Pillow |
| Numerical Processing | NumPy |
| Cloud Configuration | Azure App Configuration |
| Azure Authentication | Azure Identity / `DefaultAzureCredential` |
| Testing | Python `unittest` |
| CI | GitHub Actions |
| Setup Automation | PowerShell |

## Adding a Future Reaction
The modular reaction architecture keeps new reactions isolated:
```text
1. Add reaction media
2. Add settings to reaction_defaults.json
3. Add reaction metadata
4. Write the detector
5. Register the detector
6. Add it to REACTION_PRIORITY
7. Add unit tests
```
This keeps `main.py` focused on the real-time application loop instead of growing into a large reaction-specific `if/elif` chain.

## Design Decisions
**Personalized baseline:** Facial blendshape values vary between people, lighting conditions, and camera positions, so key reactions use change-from-neutral values instead of relying entirely on fixed raw thresholds.

**Multi-model tracking:** Face, hand, and pose models are combined because some reactions cannot be detected reliably from facial landmarks alone.

**Independent detector cadence:** Hand and pose inference run less frequently than face inference because they are more expensive and do not need frame-by-frame updates for most gestures.

**Local-first cloud integration:** Azure can tune thresholds remotely, but the application remains fully usable when Azure is unavailable.

**Testable reaction engine:** Detection rules are separated from the webcam loop so reaction logic can be tested with synthetic face, hand, and pose data.

## Status
```text
Face tracking             ✅
Hand tracking             ✅
Pose tracking             ✅
10 reaction detectors     ✅
Neutral calibration       ✅
1080p interface           ✅
Performance optimization  ✅
30 automated tests        ✅
GitHub Actions CI         ✅
Azure configuration       ✅
Local config fallback     ✅
Reaction engine modules   ✅
Windows setup automation  ✅
```

## Author
**Daniyal Tauseef**  
GitHub: [@RipLiquid](https://github.com/RipLiquid)
