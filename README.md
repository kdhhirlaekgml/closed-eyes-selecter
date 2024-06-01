# Eye State Detection Program

This program detects if a person in a given image has their eyes closed using pre-trained models for face detection, facial landmarks detection, and eye state classification.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [How It Works](#how-it-works)
- [Credits](#credits)

## Requirements

The following libraries are required to run the program:

- OpenCV
- Dlib
- TensorFlow
- NumPy

You can install these dependencies using the following commands:

```bash
pip install opencv-python dlib tensorflow numpy
```


Additionally, you will need the following files:

- `shape_predictor_68_face_landmarks.dat`: A pre-trained model for facial landmarks detection by Dlib.
- `eye_state_model.h5`: A pre-trained model for eye state classification.

## Installation

1. Clone the repository or download the project files.

```bash
git clone https://github.com/yourusername/eye-state-detection.git
cd eye-state-detection
```


2. Place the required models (`shape_predictor_68_face_landmarks.dat` and `eye_state_model.h5`) in the same directory as the script.

3. Install the necessary Python packages:

```bash
pip install -r requirements.txt
```


## Usage

1. Place the images you want to analyze in a directory.

2. Update the `image_dir` variable in the script to point to the directory containing your images.

3. Run the script:

```bash
python detect_closed_eyes.py
```


4. The program will output the paths of the images where eyes are detected to be closed.

## File Descriptions

- `detect_closed_eyes.py`: The main script that performs eye state detection.
- `shape_predictor_68_face_landmarks.dat`: The Dlib model for facial landmarks detection.
- `eye_state_model.h5`: The pre-trained model for eye state classification.

## How It Works

1. **Face Detection**: The program uses Dlib's `get_frontal_face_detector` to detect faces in the image.

2. **Facial Landmarks Detection**: For each detected face, Dlib's `shape_predictor` is used to find the coordinates of 68 facial landmarks, including those around the eyes.

3. **Eye Extraction**: The coordinates of the left and right eyes are used to extract the eye regions from the image.

4. **Eye State Classification**: Each extracted eye image is resized to 24x24 pixels, normalized, and fed into a pre-trained TensorFlow model to classify whether the eye is open or closed.

5. **Result Output**: If either eye in an image is detected as closed, the image path is added to the list of images with closed eyes, which is then printed.

## Credits

This program utilizes several pre-trained models and libraries:

- [OpenCV](https://opencv.org/): An open-source computer vision library.
- [Dlib](http://dlib.net/): A toolkit for making real-world machine learning and data analysis applications.
- [TensorFlow](https://www.tensorflow.org/): An open-source platform for machine learning.
