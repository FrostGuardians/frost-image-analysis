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
yolo_model = YOLO("best.pt")

# Path to the CSV file
CSV_FILE_PATH = "fridge_items.csv"

# Mapping detected object names to generalized categories and user-friendly names
object_name_mapping = {
    "used_banana": "banana",
    "fresh_banana": "banana",
    "open_can": "energy_drink",
    "closed_can": "energy_drink",
    "fresh_apple": "apple",
    "used_apple": "apple",
    "opened_yogurt": "yogurt",
    "closed_yogurt": "yogurt",
    # Add more mappings as needed
}

# Function to detect items using YOLOv8 and return a list of both class names and mapped item names
def detect_items_yolov8(image):
    np_img = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Perform object detection using YOLOv8
    results = yolo_model.predict(source=img, save=False)

    # Get both class names and mapped item names from YOLOv8 results
    items_info = []
    for result in results[0].boxes:
        class_name = yolo_model.names[int(result.cls[0])]  # Detected class label
        generalized_name = object_name_mapping.get(class_name, class_name)  # Map to generalized name
        print(f"Detected class: {class_name}, Mapped name: {generalized_name}")  # Debug statement
        items_info.append({"class_name": class_name, "mapped_name": generalized_name})  # Store both names

    return items_info

# Function to query GPT for expiry based on the original class name
def get_expiry_for_item(item_class, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    if item_class == "closed_can":
        item_class = "closed can of redbull"
    elif item_class == "open_can":
        item_class = "open can of redbull"
    # Query GPT for number of days using the specific class name
    query = f"Assume I just kept this '{item_class}' in a refrigerator. How many days until it will no longer be edible? Please respond with a realistic number of days."

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
        print(f"Querying GPT for expiry: {item_class}")  # Debugging line
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return extract_expiry_number(response)

    except Exception as e:
        print(f"Error querying GPT: {e}")
        return None

# Function to extract expiry number from GPT response
def extract_expiry_number(response):
    try:
        content = response.json()['choices'][0]['message']['content'].strip()
        # Extract numbers (days) from the response using regular expressions
        match = re.search(r'(\d+)', content)
        if match:
            expiry_days = int(match.group(1))  # Extract the first number found
            return expiry_days
        else:
            print(f"No valid number of days found in response: {content}")
            return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing response: {e}")
        return None

# Function to determine the category for each item
def get_category_for_item(item_class):
    item_class = item_class.lower().strip()  # Normalize the class name
    print(f"Mapping category for: {item_class}")  # Debug print

    # Ensure we're checking for both the original and mapped names
    if item_class in ["energy_drink", "closed_can", "open_can"]:
        print("Matched category: beverage")  # Debug print
        return "beverage"
    if item_class in ["banana", "used_banana", "fresh_banana"]:
        return "fruit"
    if item_class in ["apple", "fresh_apple", "used_apple"]:
        return "fruit"
    if item_class in ["yogurt", "opened_yogurt", "closed_yogurt"]:
        return "packaged_food"

    return "unknown"

# Function to update items based on detected items and the CSV data
def update_items_and_csv(detected_items, api_key):
    # Load existing CSV data
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
    else:
        df = pd.DataFrame(columns=["Name", "Expiry Date", "Category"])

    # Track updated items and query GPT for new items
    updated_items = []

    # Remove items from the CSV that are not in the new image
    new_items_mapped_names = [item["mapped_name"] for item in detected_items]
    df = df[df["Name"].isin(new_items_mapped_names)]

    # Process each detected item
    for item in detected_items:
        class_name = item["class_name"]  # Original class name (e.g., "used_apple")
        mapped_name = item["mapped_name"]  # Mapped name (e.g., "apple")

        # Check if the item already exists in CSV
        existing_row = df[df["Name"] == mapped_name]

        if existing_row.empty:
            # New item: Query GPT and add to CSV
            expiry_days = get_expiry_for_item(class_name, api_key)  # Use class name for GPT query
            if expiry_days is None:
                expiry_days = 7  # Default value if GPT fails
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime('%Y-%m-%d')
            category = get_category_for_item(class_name)

            # Add the new item to the DataFrame using pd.concat
            new_row = pd.DataFrame({"Name": [mapped_name], "Expiry Date": [expiry_date], "Category": [category]})
            df = pd.concat([df, new_row], ignore_index=True)
            updated_items.append({"Name": mapped_name, "Expiry Date": expiry_date, "Category": category})
        else:
            # Existing item: Keep the existing expiry date and category
            updated_items.append({
                "Name": existing_row["Name"].values[0],
                "Expiry Date": existing_row["Expiry Date"].values[0],
                "Category": existing_row["Category"].values[0]
            })

    # Remove duplicates based on item name to ensure items aren't listed multiple times
    df.drop_duplicates(subset=["Name"], inplace=True)

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
        detected_items = detect_items_yolov8(image)

        # Update items and CSV, only querying GPT for new or changed items
        updated_items = update_items_and_csv(detected_items, api_key)

        return jsonify({"data": updated_items}), 200

    return jsonify({"error": "File processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
