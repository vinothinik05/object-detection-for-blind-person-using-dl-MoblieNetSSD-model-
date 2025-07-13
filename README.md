 Object Detection System for Blind Persons using MobileNet SSD
An AI-powered assistive system designed to help visually impaired individuals by detecting and identifying objects in real-time, providing audio feedback for enhanced environmental awareness.

✨ Key Features:
🖼️ Real-time Object Detection: Utilizes the MobileNet SSD deep learning model for fast and accurate object recognition.
🎤 Audio Feedback: Converts detected objects into speech output to inform users.
📷 Live Video Processing: Captures and processes webcam feed continuously.
🔄 User-Friendly & Efficient: Lightweight design for potential deployment on portable devices.
📦 Pretrained Model Usage: Uses MobileNet SSD pretrained weights for reliable detection.

🛠️ Tech Stack:
Deep Learning Model: MobileNet SSD
Languages: Python
Libraries: OpenCV, NumPy, TensorFlow/PyTorch
Audio: pyttsx3 / gTTS for speech synthesis

Hardware: Webcam or camera device

🔑 API Keys
This project integrates with external services using two API keys:
OpenWeather API: To fetch real-time weather updates and environmental conditions.
OpenRoute API: To provide navigation assistance and routing information.
Note: You need to obtain your own API keys from OpenWeather and OpenRoute and add them to the project configuration before running.


📋 How to Run
Clone the repository:
git clone https://github.com/vinothinik05/object-detection-for-blind-person-using-dl-MoblieNetSSD-model-.git

Install required libraries:
pip install -r requirements.txt

Run the detection script:
python detect.py

Ensure your webcam is connected and active. The system will start detecting objects and provide audio feedback in real-time.

📁 Repository Structure:
/object-detection-for-blind-person-using-dl-MoblieNetSSD-model-
│
├── detect.py             # Main script to run object detection
├── MobileNetSSD_deploy.prototxt  # Model configuration
├── MobileNetSSD_deploy.caffemodel # Pretrained weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation


🎯 Future Improvements
Add positional audio cues (e.g., left/right object location).
optimize model for mobile and embedded systems.
Incorporate voice command controls.
Enhance detection accuracy with fine-tuned/custom models.

📧 Contact:
For questions or collaboration:
📧 vino12752@gmail.com
