import base64
import requests
import json
import re
import pandas as pd
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from config import api_key
from PIL import Image
import io
from ultralytics import YOLO

app = Flask(__name__)

# Load pre-trained YOLOv8 model for object detection
yolo_model = YOLO("best.pt")

# Function to encode the image
def encode_image(image):
    return base64.b64encode(image).decode('utf-8')

# Mapping detected object names to generalized categories
object_name_mapping = {
    "used_banana": "banana",
    "fresh_banana": "banana",
    "open_can": "can",
    "closed_can": "can",
    "fresh_apple": "apple",
    "opened_yogurt": "yogurt",
    # Add more mappings as needed
}

# Function to detect items using YOLOv8 and return bounding boxes and labels
def detect_items_yolov8(image):
    np_img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Perform object detection using YOLOv8
    results = yolo_model.predict(source=img, save=False)  # Disable save since we don't want to save the image

    # Get bounding boxes and class labels
    items_info = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        class_name = yolo_model.names[int(result.cls[0])]  # Detected class label
        generalized_name = object_name_mapping.get(class_name, class_name)  # Map to generalized name
        items_info.append({
            "class_name": generalized_name,
            "original_class": class_name,  # Store the original class for special cases (like opened or closed can)
            "bbox": (x1, y1, x2, y2)
        })

    return items_info, img

# Function to crop detected item from the original image
def crop_item_from_image(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

# Function to query GPT for expiry based on detected object
def get_expiry_for_item(item_class, original_class, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Modify the query to explicitly ask for a number of days, assuming default conditions
    query = f"Assume the '{original_class}' is now in refrigerator. How many days until it will no longer be usable or edible? Please respond with just the number of days."

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 30  # Keep it small to limit the response to a number
    }

    try:
        # Make the API request
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Debug: Print the response from GPT to check if it's valid
        print(f"GPT Response: {response.text}")

        # Extract expiry days from the response
        return extract_expiry_number(response)

    except Exception as e:
        # Log the error and return None if something goes wrong
        print(f"Error querying GPT: {e}")
        return None

# Function to extract expiry number from response using regex
def extract_expiry_number(response):
    try:
        # Parse the JSON response from GPT
        content = response.json()['choices'][0]['message']['content']

        # Debug: Log the response content
        print(f"Extracted Content: {content.strip()}")

        # Try to directly convert the content to an integer, expecting only a number
        expiry_days = int(content.strip())
        return expiry_days

    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing response: {e}")
        return None

# Function to determine the category without querying GPT
def get_category_for_item(item_class, original_class):
    # Special case for cans: Always return "beverage" for cans
    if item_class == "can":
        return "beverage"
    # Additional mappings can be added as needed
    if item_class == "banana" or item_class == "apple":
        return "fruit"
    if item_class == "yogurt":
        return "packaged_food"
    
    # Default category if nothing matches
    return "unknown"

# Function to process each detected item separately using detected class names
def process_each_detected_item(image, api_key):
    items_info, img_with_boxes = detect_items_yolov8(image)
    
    item_info_data = []
    
    for item_info in items_info:
        item_class = item_info["class_name"]
        original_class = item_info["original_class"]  # Use original class for special cases (like opened/closed can)
        bbox = item_info["bbox"]
        
        # Crop the detected item from the image
        cropped_item = crop_item_from_image(img_with_boxes, bbox)
        _, img_encoded = cv2.imencode('.jpg', cropped_item)
        cropped_image_bytes = img_encoded.tobytes()

        # Get expiry details about the item using GPT
        expiry = get_expiry_for_item(item_class, original_class, api_key)
        # Category is determined without GPT, based on predefined rules
        category = get_category_for_item(item_class, original_class)

        # If expiry is None, fallback to a default value (e.g., 365 days for cans, 7 for fresh produce)
        if expiry is None and "can" in original_class:
            expiry = 365  # Default for cans
        elif expiry is None:
            expiry = 7  # Generic fallback value for other items

        item_info_data.append({
            "Name": item_class,
            "Expiry (Days)": expiry,
            "Category": category
        })
    
    return item_info_data

# Function to store results in a DataFrame (used in upload-image endpoint)
def store_info_in_dataframe(item_info_data):
    df = pd.DataFrame(item_info_data)
    return df

# Existing '/upload-image' endpoint for processing the image and returning data
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Process the image file
        image = file.read()
        item_info_data = process_each_detected_item(image, api_key)
        df = store_info_in_dataframe(item_info_data)
        
        # Convert DataFrame to JSON for API response
        df_json = df.to_json(orient='records')
        return jsonify({"data": json.loads(df_json)}), 200

    return jsonify({"error": "File processing failed"}), 500

# Detect items and draw bounding boxes using YOLOv8 (optional functionality)
def detect_items_and_draw_boxes_yolov8(image):
    np_img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Perform object detection using YOLOv8
    results = yolo_model.predict(source=img, save=False)  # Disable save since we don't want to save the image

    # Get the annotated image with bounding boxes
    annotated_img = results[0].plot()

    return annotated_img

# New '/detect-items' endpoint for returning image with bounding boxes (YOLOv8)
@app.route('/detect-items', methods=['POST'])
def detect_items():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Process the image file
        image = file.read()

        # Detect items and draw bounding boxes using YOLOv8
        image_with_boxes = detect_items_and_draw_boxes_yolov8(image)
        
        # Convert the image with bounding boxes back to a format that can be served
        _, img_encoded = cv2.imencode('.jpg', image_with_boxes)
        img_bytes = img_encoded.tobytes()
        
        # Create a response with the image with bounding boxes
        response_img = Image.open(io.BytesIO(img_bytes))
        byte_io = io.BytesIO()
        response_img.save(byte_io, 'JPEG')
        byte_io.seek(0)

        # Send the image back with bounding boxes
        return send_file(byte_io, mimetype='image/jpeg')

    return jsonify({"error": "File processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
