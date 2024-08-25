import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('bestmodel.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize the image to match model's input shape
    img = cv2.resize(img, (28, 28))
    # Invert image if necessary
    img = cv2.bitwise_not(img)
    # Normalize the image
    img = img.astype('float32') / 255
    # Reshape for model input
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Define the region of interest (ROI) where the digit is shown
    x1, y1, x2, y2 = 100, 100, 400, 400  # Adjust ROI as necessary
    roi = frame[y1:y2, x1:x2]

    # Zoom effect
    zoom_factor = 1  # Adjust zoom factor as needed
    height, width = roi.shape[:2]
    center_x, center_y = width // 2, height // 2
    new_width, new_height = width * zoom_factor, height * zoom_factor

    # Resize ROI to zoom in
    zoomed_roi = cv2.resize(roi, (new_width, new_height))
    # Crop the zoomed ROI to the original size
    x1_crop = center_x * zoom_factor - width // 2
    y1_crop = center_y * zoom_factor - height // 2
    x2_crop = x1_crop + width
    y2_crop = y1_crop + height
    zoomed_roi = zoomed_roi[int(y1_crop):int(y2_crop), int(x1_crop):int(x2_crop)]

    # Create a black background
    black_background = np.zeros_like(frame)

    # Place the zoomed ROI on the black background
    black_background[y1:y2, x1:x2] = zoomed_roi

    # Preprocess the ROI and make a prediction
    processed_img = preprocess_image(zoomed_roi)
    prediction = model.predict(processed_img)
    digit = np.argmax(prediction)

    # Draw a rectangle around the ROI
    cv2.rectangle(black_background, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the predicted digit on the frame
    cv2.putText(black_background, f'Prediction: {digit}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with zoom effect and black background outside the ROI
    cv2.imshow('Handwritten Digit Recognition', black_background)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
