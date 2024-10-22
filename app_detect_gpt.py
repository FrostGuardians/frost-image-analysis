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
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Load pre-trained YOLOv8 model for object detection
yolo_model = YOLO("best_over.pt")

# Path to the CSV file
CSV_FILE_PATH = "fridge_items.csv"

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

# Function to detect items using YOLOv8 and return a list of mapped item names
def detect_items_yolov8(image):
    np_img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Perform object detection using YOLOv8
    results = yolo_model.predict(source=img, save=False)

    # Get mapped item names from YOLOv8 results
    items_info = []
    for result in results[0].boxes:
        class_name = yolo_model.names[int(result.cls[0])]  # Detected class label
        generalized_name = object_name_mapping.get(class_name, class_name)  # Map to generalized name
        items_info.append(generalized_name)  # Only store the mapped name

    return items_info

# Function to query GPT for expiry based on detected object
def get_expiry_for_item(item_class, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Query GPT for number of days
    query = f"Assume the '{item_class}' was recently purchased and is refrigerated. How many days until it will no longer be usable or edible? Please respond with just the number of days."

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 30
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return extract_expiry_number(response)

    except Exception as e:
        print(f"Error querying GPT: {e}")
        return None

# Function to extract expiry number from GPT response
def extract_expiry_number(response):
    try:
        content = response.json()['choices'][0]['message']['content'].strip()
        expiry_days = int(content)
        return expiry_days
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing response: {e}")
        return None

# Function to determine the category for each item
def get_category_for_item(item_class):
    if item_class == "can":
        return "beverage"
    if item_class in ["banana", "apple"]:
        return "fruit"
    if item_class == "yogurt":
        return "packaged_food"
    return "unknown"

# Function to update items based on detected items and the CSV data
def update_items_and_csv(new_items, api_key):
    # Load existing CSV data
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
    else:
        df = pd.DataFrame(columns=["Name", "Expiry Date", "Category"])

    # Track updated items and query GPT for new items
    updated_items = []

    # Remove items from the CSV that are not in the new image
    df = df[df["Name"].isin(new_items)]

    # Process each new item
    for item in new_items:
        # Check if the item already exists
        existing_row = df[df["Name"] == item]

        if existing_row.empty:
            # New item: Query GPT and add to CSV
            expiry_days = get_expiry_for_item(item, api_key)
            if expiry_days is None:
                expiry_days = 7  # Default value if GPT fails
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
            category = get_category_for_item(item)

            # Add the new item to the DataFrame using pd.concat
            new_row = pd.DataFrame({"Name": [item], "Expiry Date": [expiry_date], "Category": [category]})
            df = pd.concat([df, new_row], ignore_index=True)
            updated_items.append({"Name": item, "Expiry Date": expiry_date, "Category": category})
        else:
            # Existing item: Keep the existing expiry date and category
            updated_items.append({
                "Name": existing_row["Name"].values[0],
                "Expiry Date": existing_row["Expiry Date"].values[0],
                "Category": existing_row["Category"].values[0]
            })

    # Save the updated DataFrame back to the CSV
    df.to_csv(CSV_FILE_PATH, index=False)

    return updated_items

# '/upload-image' endpoint for processing the image and returning updated data
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
        new_items = detect_items_yolov8(image)

        # Update items and CSV, only querying GPT for new or changed items
        updated_items = update_items_and_csv(new_items, api_key)

        return jsonify({"data": updated_items}), 200

    return jsonify({"error": "File processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)