# 💖 AI Hopeless Romantic

An interactive computer vision application that detects when two hands form a **heart gesture** together. Once recognized, it beautifully renders an overlay heart and plays a romantic tune straight on your desktop!

---

## ✨ Features

- **Two-Hand Gesture Recognition:** Accurately maps both hands using MediaPipe's `HandLandmarker` model to verify index and thumb tip alignment.
- **Visual Feedback:** Renders a glowing red heart overlay right where your hands meet, complete with a live status message.
- **Romantic Audio/Video Trigger:** Automatically triggers a romantic video clip/audio playback (`so-easy-to-fall-in-love---olivia-dean.mp4`) via system subprocess when the gesture is successfully held.
- **Smooth Hold & Release Logic:** Features a buffer timer (`HOLD_DURATION`) so the effect remains stable and doesn't flicker.

---

## 🛠️ Prerequisites & Dependencies

Make sure you have Python installed, then run the following command to install the necessary packages:

```bash
pip install opencv-python mediapipe numpy
