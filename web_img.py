import os
import pandas as pd
import json
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify, Response
import urllib.parse

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'

# Ensure static folder exists
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# Read CSV data
def get_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_submission.csv')
    df = pd.read_csv(csv_path)
    return df

# Get all image files from train directory and subdirectories with folder information
def get_image_paths():
    train_dir = os.path.join(os.path.dirname(__file__), 'train')
    image_files = []
    category_images = {}
    
    # Check if train directory exists
    if not os.path.exists(train_dir):
        return [], {}
    
    # Walk through all subdirectories and find image files
    for root, _, files in os.walk(train_dir):
        category = os.path.basename(root)
        if category == "train":  # Skip the root train folder
            continue
            
        # Initialize category array if not exists
        if category not in category_images:
            category_images[category] = []
            
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Use forward slashes for URL paths
                rel_dir = os.path.relpath(root, os.path.dirname(__file__)).replace('\\', '/')
                rel_path = f"{rel_dir}/{file}"
                image_id = file.split('.')[0]  # Extract ID from filename without extension
                
                image_info = {
                    'id': image_id,
                    'path': rel_path,
                    'filename': file,
                    'category': category
                }
                
                image_files.append(image_info)
                category_images[category].append(image_info)
    
    return image_files, category_images

# Get existing classifications if available
def get_classifications():
    classification_path = os.path.join(os.path.dirname(__file__), 'classifications.json')
    if os.path.exists(classification_path):
        try:
            with open(classification_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Get available classes
def get_available_classes():
    classes_path = os.path.join(os.path.dirname(__file__), 'available_classes.json')
    if os.path.exists(classes_path):
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return ["0"]  # Default class
    return ["0"]  # Default class

@app.route('/')
def index():
    df = get_data()
    # Get unique types for filter options
    types = sorted(df['type'].unique())
    
    # Get filter parameter or show all if not specified
    selected_type = request.args.get('type', None)
    selected_category = request.args.get('category', None)
    
    if selected_type is not None:
        try:
            selected_type = int(selected_type)
            filtered_df = df[df['type'] == selected_type]
        except ValueError:
            filtered_df = df
    else:
        filtered_df = df
    
    # Get all image paths and category information
    image_files, category_images = get_image_paths()
    
    # Convert to list of dictionaries for template
    images = filtered_df.to_dict('records')
    
    # Get all categories (folders)
    categories = sorted(list(category_images.keys()))
    
    # Get any existing classifications
    classifications = get_classifications()
    
    # Get available classes
    available_classes = get_available_classes()
    
    return render_template('index.html', 
                          images=images, 
                          types=types, 
                          selected_type=selected_type,
                          selected_category=selected_category,
                          image_files=image_files,
                          categories=categories,
                          category_images=category_images,
                          classifications=classifications,
                          available_classes=available_classes)

@app.route('/image/<path:filename>')
def serve_image(filename):
    try:
        # URL decode the filename to handle special characters
        decoded_filename = urllib.parse.unquote(filename)
        
        # Normalize slashes for the OS
        if os.name == 'nt':  # Windows
            decoded_filename = decoded_filename.replace('/', '\\')
        
        # Get the absolute path of the file
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, decoded_filename)
        
        # Check if file exists
        if not os.path.isfile(file_path):
            # Return a placeholder image if the file doesn't exist
            placeholder_path = os.path.join(base_dir, 'static', 'placeholder.png')
            
            # Create placeholder if it doesn't exist
            if not os.path.exists(placeholder_path):
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (200, 150), color=(240, 240, 240))
                d = ImageDraw.Draw(img)
                d.text((40, 65), "Image Not Found", fill=(0, 0, 0))
                
                # Ensure static directory exists
                os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
                img.save(placeholder_path)
                
            return send_from_directory(os.path.join(base_dir, 'static'), 'placeholder.png')
        
        # If file exists, extract directory and filename to use with send_from_directory
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Debug information
        print(f"Serving file: {file_path}")
        print(f"Directory: {directory}")
        print(f"Filename: {filename}")
        
        return send_from_directory(directory, filename)
        
    except Exception as e:
        print(f"Error serving image: {e}")
        # Return placeholder on error
        return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'placeholder.png')

@app.route('/save_classifications', methods=['POST'])
def save_classifications():
    try:
        data = request.json
        classification_path = os.path.join(os.path.dirname(__file__), 'classifications.json')
        
        # Save as JSON for easy retrieval
        with open(classification_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return jsonify({"status": "success", "message": "Classifications saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_classes', methods=['POST'])
def save_classes():
    try:
        data = request.json
        classes_path = os.path.join(os.path.dirname(__file__), 'available_classes.json')
        
        # Save as JSON for easy retrieval
        with open(classes_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return jsonify({"status": "success", "message": "Classes saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/export_classifications')
def export_classifications():
    try:
        # Get classifications from JSON file
        classification_path = os.path.join(os.path.dirname(__file__), 'classifications.json')
        if not os.path.exists(classification_path):
            return jsonify({"status": "error", "message": "No classifications found"})
            
        with open(classification_path, 'r', encoding='utf-8') as f:
            classifications = json.load(f)
        
        # Read CSV for original classes
        df = get_data()
        id_to_type = {str(row['id']): row['type'] for index, row in df.iterrows()}
        
        # Create export content
        export_content = "image name|class|new class\n"
        for image_id, new_class in classifications.items():
            original_class = id_to_type.get(image_id, "")
            export_content += f"{image_id}|{original_class}|{new_class}\n"
        
        # Create response with file download
        response = Response(
            export_content,
            mimetype="text/plain",
            headers={"Content-disposition": "attachment; filename=classifications_export.txt"}
        )
        return response
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/upload_classifications', methods=['POST'])
def upload_classifications():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})
            
        if file:
            try:
                content = file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                # Skip header
                header = lines[0]
                if not header.startswith('image name|'):
                    return jsonify({"status": "error", "message": "Invalid file format"})
                
                # Create dictionary from the uploaded data
                classifications = {}
                for i in range(1, len(lines)):
                    parts = lines[i].split('|')
                    if len(parts) >= 3:
                        image_id = parts[0].strip()
                        new_class = parts[2].strip()
                        classifications[image_id] = new_class
                
                # Save the imported classifications
                classification_path = os.path.join(os.path.dirname(__file__), 'classifications.json')
                with open(classification_path, 'w', encoding='utf-8') as f:
                    json.dump(classifications, f, ensure_ascii=False, indent=2)
                    
                return jsonify({"status": "success", "message": "Classifications imported successfully"})
                    
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error parsing file: {str(e)}"})
                
        return jsonify({"status": "error", "message": "Unknown error"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
