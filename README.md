# PeakPlan

Climbing-analysis demo: pose estimation → center of mass / base of support / movement metrics → annotated MP4 clips and a Hebrew (RTL) HTML dashboard.

## Contents

| File | Description |
| --- | --- |
| `climbing_demo.html` | Dashboard (Hebrew, RTL). Open in any browser; videos and images are embedded, no external paths. Pages: Overview, Landing accuracy, Momentum, Distance from wall, Attempt comparison. |
| `attempt1.mp4` – `attempt3.mp4` | Annotated clips (segments 0–8.2s, 8.2–16.5s, 16.5–24.7s). |
| `metrics.json` | All raw values plus 0–100 scores per segment, and the reference segment. |
| `climb_demo.py` | The script that produces everything above. |

## Re-running the analysis

Requirements: Python 3.12 and `ffmpeg` on your `PATH`.

```bash
pip install mediapipe==0.10.21 opencv-python matplotlib
```

One climb, split into 3 segments (zoom-follow crop):

```bash
python climb_demo.py out --single "your_video.mp4"
```

Three separate attempts:

```bash
python climb_demo.py out a1.mp4 a2.mp4 a3.mp4
```
