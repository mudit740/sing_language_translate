from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import io
import base64

model = keras.models.load_model('asl_cnn_model.h5')

class_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'DELETE',   
    27: 'NOTHING',
    28: 'SPACE'
}

app = Flask(__name__)

def remove_background(image_data):
    # Convert base64 image data to OpenCV format
    image_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(image_data))
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve the segmentation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to create a binary mask
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area (assuming it's the hand)
    max_contour = max(contours, key=cv2.contourArea)

    # Create an empty mask and draw the largest contour on it
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [max_contour], 0, 255, thickness=cv2.FILLED)

    # Bitwise AND the original image with the mask to keep only the hand
    result = cv2.bitwise_and(cv_img, cv_img, mask=mask)

    return result

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    img_data = request.form['image']

    # Remove the background
    img_with_hand = remove_background(img_data)

    # Preprocess the image
    img_with_hand = cv2.resize(img_with_hand, (64, 64))
    img_with_hand = cv2.cvtColor(img_with_hand, cv2.COLOR_BGR2GRAY)
    img_with_hand = img_with_hand.reshape((1, 64, 64, 1))
    img_with_hand = img_with_hand / 255.0

    # Make the prediction
    prediction = model.predict(img_with_hand)
    prediction = np.argmax(prediction)

    # Convert the prediction to a character using the class_dict
    prediction_char = class_dict[prediction]

    # Return the result as a JSON object
    return jsonify({'prediction': prediction_char})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

