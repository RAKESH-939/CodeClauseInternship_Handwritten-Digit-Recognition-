# CodeClauseInternship_Handwritten-Digit-Recognition-

## Handwritten Digit Recognition with Webcam
This project uses a trained Convolutional Neural Network (CNN) model to recognize handwritten digits in real-time using a webcam. The model is trained on the MNIST dataset and can predict digits drawn within a specified region on the webcam feed.

##Features
Real-time Digit Recognition: Recognizes handwritten digits in real-time from a webcam feed.
Region of Interest (ROI): The area where you can draw digits is highlighted, and only this area is processed for digit recognition.
Zoom Effect: The region of interest is zoomed for better visibility before being processed by the model.
Black Background: The area outside the ROI is blacked out to focus on the digit recognition area.

## Download the Trained Model:

Ensure you have the trained model (bestmodel.h5) in the project directory. If you don't have the model, you can train one using the MNIST dataset or download it from the repository if provided.

## Run the Application:

bash
python app.py

## Usage

### 1.Launch the Application:
When you run the application, it will start capturing video from your webcam.

### 2.Drawing Digits:
Draw a digit within the green rectangle (ROI) in the webcam feed. The digit should be clearly visible and fit within the rectangle.

### 3.Prediction:
The application will display the predicted digit above the ROI in real-time.

### 4.Exit:
Press the q key to exit the application.

## How It Works

Model: The project uses a pre-trained CNN model (bestmodel.h5) built on the MNIST dataset. The model can recognize digits from 0 to 9.

Preprocessing: The captured frame within the ROI is converted to grayscale, resized to 28x28 pixels, inverted (since MNIST digits are white on a black background), normalized, and then passed to the model for prediction.

Zoom Effect: The ROI is zoomed for better accuracy before preprocessing.

## Project Structure
plaintext

handwritten-digit-recognition/
│
├── bestmodel.h5              # Trained model file
├── recognize_digits.py       # Main script to run the application
└── README.md                 # Project documentation

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
. Keras and TensorFlow for providing powerful deep learning libraries.
. OpenCV for real-time computer vision capabilities.
. The MNIST Dataset for digit training data.
