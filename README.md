# Crowd Detection and Wait-time Predictions

This project demonstrates real-time video processing using the YOLOv8 object detection model to count people and simulate POS (Point of Sale) operations. The application features an interactive front-end to select video streams from multiple outlets and provides live updates on the detected people count, bill simulations, and estimated wait times.

## Features

- **Real-time Video Analysis**: Detect and count people in video streams using the YOLOv8 model.
- **POS Simulation**: Simulates bill processing for a food outlet with dynamic wait time estimation.
- **Interactive UI**: Users can select between different outlets and monitor the results in real-time.
- **Flask + SocketIO Integration**: Seamless communication between the back-end and front-end for real-time updates.
- **Themed UI Design**: Modern, responsive interface with a dark mode theme.

---

## Getting Started

Follow these steps to set up and run the application on your local machine.

### Prerequisites

- **Python 3.8 or later**
- Virtual environment (recommended)
- A YOLOv8 model file (`yolov8n.pt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/live-yolov8-pos.git
   cd live-yolov8-pos
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up `pyngrok` for public access (optional):
   - Install ngrok and set up your API key:
     ```python
     from pyngrok import ngrok
     ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
     ```

4. Ensure the video files exist in the specified paths:
   - Update the `video_paths` dictionary in `app.py` with your video file paths.

---

## Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Access the application:
   - Locally: `http://127.0.0.1:5000`
   - Via ngrok (if configured): Check the terminal output for the public URL.

---

## Usage

1. Open the application in your web browser.
2. Select an outlet from the homepage to start processing its video feed.
3. Monitor:
   - **People Count**: Real-time count of detected individuals.
   - **POS Simulation**: Bill count and estimated wait time.
4. Return to the homepage to select a different outlet.

---

## Technical Details

### YOLOv8 Model

- This project uses the YOLOv8 model for object detection.
- The model is initialized with:
  ```python
  model = YOLO('yolov8n.pt')
  ```

### POS Simulation

- The POS system simulates bill processing, where:
  - **Base Wait Time**: `2 minutes per person`
  - **Additional Time**: `1.5 minutes per bill`
  - **Food Preparation Time**: `5 minutes`

### Flask-SocketIO

- Used for real-time communication between the server and the front-end to push frame updates and POS data.

---

## Customization

1. **Video Streams**:
   - Update the `video_paths` dictionary in `app.py` to point to your video files.
2. **Wait Time Calculation**:
   - Modify constants `wait_time_per_person`, `additional_time_per_bill`, and `food_prep_time`.

---

## Requirements

Create a `requirements.txt` file to manage dependencies:

```text
Flask
Flask-SocketIO
numpy
opencv-python
pyngrok
ultralytics
```

