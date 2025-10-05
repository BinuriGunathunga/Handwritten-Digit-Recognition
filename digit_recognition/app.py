from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io
import cv2

app = Flask(__name__)

# Load the trained model
print("Loading model...")
model = keras.models.load_model('digit_recognition_model.h5')
print("Model loaded successfully!")

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    # Convert to grayscale
    img = image.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert colors if needed (MNIST has white digits on black background)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def segment_math_expression(image):
    """Segment math expression image into individual characters"""
    # Convert PIL to numpy array
    img_array = np.array(image.convert('L'))
    
    # Invert if needed (we want black text on white background for processing)
    if np.mean(img_array) < 127:
        img_array = 255 - img_array
    
    # Apply thresholding
    _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes and sort by x-coordinate (left to right)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small contours (noise)
        if w > 10 and h > 10:  # Increased minimum size
            bounding_boxes.append((x, y, w, h))
    
    # Sort by x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    # Calculate median height and width to distinguish digits from operators
    if len(bounding_boxes) > 0:
        heights = [h for (x, y, w, h) in bounding_boxes]
        widths = [w for (x, y, w, h) in bounding_boxes]
        median_height = np.median(heights)
        median_width = np.median(widths)
    else:
        median_height = 0
        median_width = 0
    
    # Extract individual characters
    segments = []
    for (x, y, w, h) in bounding_boxes:
        # Add some padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img_array.shape[1], x + w + padding)
        y_end = min(img_array.shape[0], y + h + padding)
        
        char_img = img_array[y_start:y_end, x_start:x_end]
        
        # Determine if likely an operator based on size
        is_likely_operator = False
        if median_height > 0:
            height_ratio = h / median_height
            width_ratio = w / median_width if median_width > 0 else 1
            
            # Operators are usually smaller or have different proportions
            if height_ratio < 0.7 or (w > h * 1.5):  # Much wider than tall
                is_likely_operator = True
        
        segments.append({
            'image': char_img,
            'position': x,
            'box': (x, y, w, h),
            'is_likely_operator': is_likely_operator,
            'aspect_ratio': w / h if h > 0 else 1
        })
    
    return segments

def classify_character(char_img):
    """Classify a character as digit or operator"""
    # Resize to 28x28
    char_pil = Image.fromarray(char_img)
    char_pil = char_pil.resize((28, 28))
    
    # Convert to array
    char_array = np.array(char_pil)
    
    # Check for operator patterns BEFORE digit classification
    height, width = char_img.shape
    
    # Operator detection based on shape analysis
    operator = detect_operator(char_img)
    if operator:
        return {
            'is_digit': False,
            'value': operator,
            'confidence': 95.0,
            'is_operator': True
        }
    
    # If not an operator, classify as digit
    # Invert (MNIST has white digits on black)
    char_array = 255 - char_array
    
    # Normalize
    char_array = char_array.astype('float32') / 255.0
    char_array = np.expand_dims(char_array, axis=0)
    char_array = np.expand_dims(char_array, axis=-1)
    
    # Predict
    prediction = model.predict(char_array, verbose=0)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    
    return {
        'is_digit': True,
        'value': str(predicted_digit),
        'confidence': confidence,
        'is_operator': False
    }

def detect_operator(char_img):
    """Detect mathematical operators using shape analysis"""
    height, width = char_img.shape
    
    if height == 0 or width == 0:
        return None
    
    aspect_ratio = width / height
    
    # Calculate non-zero pixels (ink pixels)
    total_pixels = height * width
    ink_pixels = np.sum(char_img > 100)
    ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0
    
    # Plus sign detection (+)
    # Plus has roughly equal width and height, cross pattern in center
    if 0.6 < aspect_ratio < 1.4 and 0.15 < ink_ratio < 0.4:
        # Check for horizontal and vertical lines
        center_h = height // 2
        center_w = width // 2
        
        # Check horizontal line through center
        h_line = char_img[center_h-2:center_h+2, :]
        h_ink = np.sum(h_line > 100) / (4 * width) if width > 0 else 0
        
        # Check vertical line through center
        v_line = char_img[:, center_w-2:center_w+2]
        v_ink = np.sum(v_line > 100) / (height * 4) if height > 0 else 0
        
        if h_ink > 0.4 and v_ink > 0.4:
            return '+'
    
    # Minus sign detection (-)
    # Minus is typically horizontal, wide and short
    if aspect_ratio > 1.5 and 0.1 < ink_ratio < 0.3:
        # Check if ink is mostly in the middle horizontal band
        middle_band = char_img[height//3:2*height//3, :]
        middle_ink = np.sum(middle_band > 100)
        if middle_ink / ink_pixels > 0.7 if ink_pixels > 0 else False:
            return '-'
    
    # Multiplication sign detection (x or ×)
    # X shape with diagonal lines
    if 0.6 < aspect_ratio < 1.4 and 0.15 < ink_ratio < 0.4:
        # Check diagonal patterns
        diag1_count = 0
        diag2_count = 0
        
        for i in range(min(height, width)):
            # Main diagonal (top-left to bottom-right)
            if i < height and i < width:
                if char_img[i, i] > 100:
                    diag1_count += 1
            
            # Anti-diagonal (top-right to bottom-left)
            if i < height and (width - 1 - i) >= 0:
                if char_img[i, width - 1 - i] > 100:
                    diag2_count += 1
        
        diag_threshold = min(height, width) * 0.3
        if diag1_count > diag_threshold and diag2_count > diag_threshold:
            return 'x'
    
    # Division sign detection (÷)
    # Usually has a horizontal line with dots above and below
    if 0.6 < aspect_ratio < 1.4 and 0.15 < ink_ratio < 0.35:
        top_third = char_img[0:height//3, :]
        middle_third = char_img[height//3:2*height//3, :]
        bottom_third = char_img[2*height//3:, :]
        
        top_ink = np.sum(top_third > 100)
        middle_ink = np.sum(middle_third > 100)
        bottom_ink = np.sum(bottom_third > 100)
        
        # Division has ink in top, middle, and bottom
        if top_ink > 0 and middle_ink > top_ink and bottom_ink > 0:
            return '÷'
    
    return None

def detect_operator_simple(segments):
    """Simple operator detection based on segment properties"""
    expression_parts = []
    
    for i, seg in enumerate(segments):
        char_result = classify_character(seg['image'])
        expression_parts.append(char_result)
    
    return expression_parts

def parse_and_solve(expression_parts):
    """Parse expression and solve"""
    expression_str = ''.join([part['value'] for part in expression_parts])
    
    # Replace operators for eval
    expression_eval = expression_str.replace('x', '*').replace('÷', '/').replace('×', '*')
    
    try:
        result = eval(expression_eval)
        return {
            'expression': expression_str,
            'result': result,
            'success': True
        }
    except:
        return {
            'expression': expression_str,
            'result': None,
            'success': False,
            'error': 'Could not parse expression'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        
        # Get all probabilities
        all_probabilities = {
            str(i): float(prediction[0][i] * 100) 
            for i in range(10)
        }
        
        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/solve-math', methods=['POST'])
def solve_math():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        image = Image.open(file.stream)
        
        # Segment the image into individual characters
        segments = segment_math_expression(image)
        
        if len(segments) == 0:
            return jsonify({'error': 'No characters detected in image'}), 400
        
        # Classify each segment
        expression_parts = detect_operator_simple(segments)
        
        # Build expression string and get details
        detected_chars = []
        for part in expression_parts:
            detected_chars.append({
                'character': part['value'],
                'confidence': round(part['confidence'], 2),
                'type': 'operator' if part['is_operator'] else 'digit'
            })
        
        # Parse and solve
        solution = parse_and_solve(expression_parts)
        
        return jsonify({
            'detected_characters': detected_chars,
            'expression': solution['expression'],
            'result': solution['result'],
            'success': solution['success'],
            'error': solution.get('error', None)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)