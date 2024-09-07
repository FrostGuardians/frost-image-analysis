import base64
import requests
import json
import re
import pandas as pd
from flask import Flask, request, jsonify
from config import api_key


app = Flask(__name__)

# Function to encode the image
def encode_image(image):
    return base64.b64encode(image).decode('utf-8')

# Function to get food items from image
def get_food_items_from_image(image, api_key):
    base64_image = encode_image(image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image, give a list of items?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return extract_items_from_response(response)

# Function to extract items from response
def extract_items_from_response(response):
    try:
        content = response.json()['choices'][0]['message']['content']
        lines = content.split('\n')
        items = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if re.match(r'^\d+\.\s+', line)]
        return items
    except (KeyError, IndexError) as e:
        print("Error processing response:", e)
        return []

# Function to get expiry for each item
def get_expiry_for_item(item, image, api_key):
    base64_image = encode_image(image)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"How many days until this item '{item}' won't be eatable anymore? Give just one number."
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return extract_expiry_number(response)

# Function to get category for each item
def get_category_for_item(item, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"Is this item '{item}' a packaged food, fruit, or sauce? Just answer with one word."
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return standardize_category_response(extract_category_from_response(response))

# Function to extract expiry number from response
def extract_expiry_number(response):
    try:
        content = response.json()['choices'][0]['message']['content']
        return int(content.strip())
    except (KeyError, IndexError, ValueError) as e:
        print("Error processing response:", e)
        return None

# Function to extract and standardize category from response
def extract_category_from_response(response):
    try:
        content = response.json()['choices'][0]['message']['content']
        return content.strip().lower()
    except (KeyError, IndexError) as e:
        print("Error processing response:", e)
        return "unknown"

# Function to standardize the category response
def standardize_category_response(category):
    if "packaged" in category:
        return "packaged_food"
    elif "fruit" in category:
        return "fruit"
    elif "sauce" in category:
        return "sauce"
    else:
        return "unknown"

# Function to process image and get items, expiry, and category for each item
def process_image_and_get_info(image, api_key):
    items = get_food_items_from_image(image, api_key)
    
    item_info_data = []
    for item in items:
        expiry = get_expiry_for_item(item, image, api_key)
        category = get_category_for_item(item, api_key)
        item_info_data.append({
            "Name": item,          # Renamed from "Items" to "Type"
            "Expiry (Days)": expiry,
            "Category": category
        })
    
    return item_info_data

# Function to store results in a DataFrame
def store_info_in_dataframe(item_info_data):
    df = pd.DataFrame(item_info_data)
    return df

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
        item_info_data = process_image_and_get_info(image, api_key)
        df = store_info_in_dataframe(item_info_data)
        
        # Convert DataFrame to JSON for API response
        df_json = df.to_json(orient='records')
        return jsonify({"data": json.loads(df_json)}), 200

    return jsonify({"error": "File processing failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
