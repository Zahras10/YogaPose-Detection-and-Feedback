Yoga Pose Detection and Feedback Generation System using Mediapipe.

This project uses MediaPipe, OpenCV, and scikit-learn to detect yoga poses in real time and provide
instant feedback on posture accuracy. It features a GUI-based interface and uses landmark-based
machine learning models for pose classification.

Features:
- Real-time yoga pose detection via webcam
- Instant feedback on posture correctness
- Pretrained models for multiple yoga poses
- User-friendly GUI (built with Tkinter)
- Machine learning-based pose classification using body landmarks.

- Project Structure:
fityoga/
- main/   -> All core Python scripts (GUI, pose processing, training)
- models/ -> Pretrained ML models (.pkl files)
- data/   -> CSV datasets of landmark coordinates
- images/ -> Reference pose images for GUI/README- requirements.txt
- README.md
- .gitignore


Technologies Used:
- Python 3.x
- MediaPipe
- OpenCV
- scikit-learn
- NumPy
- Tkinter
- Pillow


How to Run the Project:
1. Install dependencies:
pip install -r requirements.txt

2. Run the main GUI:
cd main
python GUI_Master.py

License:
This project is released under the MIT License.


About Me:
This project was created as part of my academic work in computer vision and human posture
analysis.
Feel free to reach out if you'd like to collaborate or have feedback!
