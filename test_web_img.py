# Web-based image viewer for test folder images

from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

# Constants
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
DEFAULT_TEST_FOLDER = os.path.join('d:', 'project', 'oai', 'test_images')

app = Flask(__name__)

def get_images_from_folder(folder_path):
    """Get all image files from a folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(list(folder.glob(f"*{ext}")))
        image_paths.extend(list(folder.glob(f"*{ext.upper()}")))
    
    # Sort for consistent ordering
    return sorted(image_paths)

def get_image_metadata(image_path):
    """Get basic metadata for an image."""
    try:
        with Image.open(image_path) as img:
            return {
                'filename': os.path.basename(image_path),
                'path': str(image_path),
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            }
    except Exception as e:
        print(f"Error getting metadata for {image_path}: {e}")
        return {
            'filename': os.path.basename(image_path),
            'path': str(image_path),
            'error': str(e)
        }

def image_to_base64(image_path, max_width=800):
    """Convert image to base64 for embedding in HTML."""
    try:
        with Image.open(image_path) as img:
            # Resize if needed
            w, h = img.size
            if w > max_width:
                ratio = max_width / w
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format=img.format if img.format else 'JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/{img.format.lower() if img.format else 'jpeg'};base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    """Home page - display folders that can be browsed."""
    image_folder = request.args.get('folder', DEFAULT_TEST_FOLDER)
    
    # Create test folder if it doesn't exist
    if image_folder == DEFAULT_TEST_FOLDER and not os.path.exists(DEFAULT_TEST_FOLDER):
        os.makedirs(DEFAULT_TEST_FOLDER)
        print(f"Created test folder: {DEFAULT_TEST_FOLDER}")
    
    return render_template('index.html', 
                          folder=image_folder, 
                          default_folder=DEFAULT_TEST_FOLDER)

@app.route('/browse')
def browse():
    """API endpoint to get images in a folder."""
    folder = request.args.get('folder', DEFAULT_TEST_FOLDER)
    limit = request.args.get('limit', None)
    grid_view = request.args.get('grid', 'true').lower() == 'true'
    
    if limit:
        try:
            limit = int(limit)
        except ValueError:
            limit = None
    
    # Get image paths
    image_paths = get_images_from_folder(folder)
    
    # Apply limit if specified
    if limit:
        image_paths = image_paths[:limit]
    
    # Get metadata for each image
    images = []
    for path in image_paths:
        metadata = get_image_metadata(path)
        images.append(metadata)
    
    return jsonify({
        'folder': folder,
        'grid_view': grid_view,
        'count': len(images),
        'images': images
    })

@app.route('/image/<path:filename>')
def serve_image(filename):
    """Serve an image file."""
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    return send_from_directory(directory, basename)

@app.route('/view/<path:image_path>')
def view_image(image_path):
    """View a single image with details."""
    # Normalize path
    image_path = os.path.normpath(image_path)
    
    # Get metadata
    metadata = get_image_metadata(image_path)
    
    # Get base64 representation
    image_data = image_to_base64(image_path)
    
    return render_template('image_view.html', 
                          image=metadata,
                          image_data=image_data)

@app.route('/grid')
def grid():
    """Display images in a grid view."""
    folder = request.args.get('folder', DEFAULT_TEST_FOLDER)
    limit = request.args.get('limit', None)
    
    if limit:
        try:
            limit = int(limit)
        except ValueError:
            limit = None
    
    return render_template('grid.html', 
                          folder=folder,
                          limit=limit)

# Create templates directory and files if they don't exist
def create_templates():
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html
    with open(os.path.join(templates_dir, 'test_index.html'), 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Image Viewer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { margin-bottom: 20px; }
        .controls { margin-bottom: 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .image-card { border: 1px solid #ddd; border-radius: 5px; padding: 10px; transition: transform 0.3s; }
        .image-card:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .image-card img { width: 100%; height: 200px; object-fit: cover; border-radius: 3px; }
        .image-info { margin-top: 10px; font-size: 14px; }
        .loader { text-align: center; padding: 20px; }
        .view-toggle { margin-bottom: 15px; }
        .btn { padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #45a049; }
        input[type="text"] { padding: 8px; width: 100%; max-width: 500px; }
        .error { color: red; padding: 10px; background: #ffeeee; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Image Viewer</h1>
            <p>Browse and view images in the test folder</p>
        </div>

        <div class="controls">
            <div>
                <label for="folder">Folder path:</label>
                <input type="text" id="folder" value="{{ folder }}" style="width: 70%;">
                <button class="btn" onclick="loadImages()">Browse</button>
            </div>
            <div style="margin-top: 10px;">
                <label for="limit">Limit:</label>
                <input type="number" id="limit" value="20" style="width: 80px;">
                
                <label style="margin-left: 15px;">View:</label>
                <label><input type="radio" name="view" value="grid" checked> Grid</label>
                <label><input type="radio" name="view" value="list"> List</label>
            </div>
        </div>

        <div class="view-toggle">
            <button class="btn" id="viewModeBtn" onclick="toggleViewMode()">Switch to List View</button>
        </div>

        <div id="error" class="error" style="display: none;"></div>
        <div id="loader" class="loader" style="display: none;">Loading images...</div>
        <div id="imageGrid" class="image-grid"></div>
        <div id="imageList" style="display: none;"></div>
    </div>

    <script>
        // Load images when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            loadImages();
        });

        let currentViewMode = 'grid';

        // Toggle between grid and list views
        function toggleViewMode() {
            const gridView = document.getElementById('imageGrid');
            const listView = document.getElementById('imageList');
            const viewBtn = document.getElementById('viewModeBtn');
            
            if (currentViewMode === 'grid') {
                gridView.style.display = 'none';
                listView.style.display = 'block';
                viewBtn.textContent = 'Switch to Grid View';
                currentViewMode = 'list';
            } else {
                gridView.style.display = 'grid';
                listView.style.display = 'none';
                viewBtn.textContent = 'Switch to List View';
                currentViewMode = 'grid';
            }
        }

        // Load images from the specified folder
        function loadImages() {
            const folder = document.getElementById('folder').value;
            const limit = document.getElementById('limit').value;
            const gridView = document.querySelector('input[name="view"][value="grid"]').checked;
            
            document.getElementById('loader').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            document.getElementById('imageGrid').innerHTML = '';
            document.getElementById('imageList').innerHTML = '';
            
            fetch(`/browse?folder=${encodeURIComponent(folder)}&limit=${limit}&grid=${gridView}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('loader').style.display = 'none';
                    
                    if (data.images.length === 0) {
                        showError(`No images found in ${folder}`);
                        return;
                    }
                    
                    // Populate grid view
                    const gridContainer = document.getElementById('imageGrid');
                    data.images.forEach(image => {
                        const card = document.createElement('div');
                        card.className = 'image-card';
                        
                        card.innerHTML = `
                            <a href="/view/${encodeURIComponent(image.path)}" target="_blank">
                                <img src="/image/${encodeURIComponent(image.path)}" alt="${image.filename}">
                            </a>
                            <div class="image-info">
                                <div><strong>${image.filename}</strong></div>
                                <div>${image.width} × ${image.height} px</div>
                            </div>
                        `;
                        gridContainer.appendChild(card);
                    });
                    
                    // Populate list view
                    const listContainer = document.getElementById('imageList');
                    const table = document.createElement('table');
                    table.style.width = '100%';
                    table.style.borderCollapse = 'collapse';
                    
                    table.innerHTML = `
                        <thead>
                            <tr style="background: #f5f5f5;">
                                <th style="text-align: left; padding: 8px;">Preview</th>
                                <th style="text-align: left; padding: 8px;">Filename</th>
                                <th style="text-align: left; padding: 8px;">Dimensions</th>
                                <th style="text-align: left; padding: 8px;">Format</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.images.map(image => `
                                <tr style="border-bottom: 1px solid #ddd;">
                                    <td style="padding: 8px;">
                                        <a href="/view/${encodeURIComponent(image.path)}" target="_blank">
                                            <img src="/image/${encodeURIComponent(image.path)}" alt="${image.filename}" style="width: 100px; height: 80px; object-fit: cover;">
                                        </a>
                                    </td>
                                    <td style="padding: 8px;">${image.filename}</td>
                                    <td style="padding: 8px;">${image.width} × ${image.height}</td>
                                    <td style="padding: 8px;">${image.format}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    `;
                    listContainer.appendChild(table);
                })
                .catch(error => {
                    document.getElementById('loader').style.display = 'none';
                    showError(`Error loading images: ${error}`);
                });
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
    </script>
</body>
</html>''')
    
    # Create image_view.html - Fix the encoding issue with the arrow character
    with open(os.path.join(templates_dir, 'image_view.html'), 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>{{ image.filename }} - Image Viewer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { margin-bottom: 20px; }
        .image-container { text-align: center; margin-bottom: 20px; }
        .image-container img { max-width: 100%; max-height: 80vh; }
        .metadata { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .metadata table { width: 100%; border-collapse: collapse; }
        .metadata td, .metadata th { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .metadata th { width: 120px; }
        .back-button { margin-bottom: 20px; }
        .btn { padding: 8px 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ image.filename }}</h1>
            <div class="back-button">
                <a href="javascript:history.back()" class="btn">&larr; Back to gallery</a>
            </div>
        </div>

        <div class="image-container">
            <img src="{{ image_data }}" alt="{{ image.filename }}">
        </div>

        <div class="metadata">
            <h2>Image Metadata</h2>
            <table>
                <tr>
                    <th>Filename</th>
                    <td>{{ image.filename }}</td>
                </tr>
                <tr>
                    <th>Path</th>
                    <td>{{ image.path }}</td>
                </tr>
                <tr>
                    <th>Dimensions</th>
                    <td>{{ image.width }} × {{ image.height }} px</td>
                </tr>
                <tr>
                    <th>Format</th>
                    <td>{{ image.format }}</td>
                </tr>
                <tr>
                    <th>Mode</th>
                    <td>{{ image.mode }}</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>''')
    
    # Create grid.html - Also use UTF-8 encoding for consistency
    with open(os.path.join(templates_dir, 'grid.html'), 'w', encoding='utf-8') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Image Grid View</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { margin-bottom: 20px; }
        .controls { margin-bottom: 20px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .image-item { border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }
        .image-item img { width: 100%; height: 200px; object-fit: cover; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Image Grid View</h1>
            <p>Folder: {{ folder }}</p>
        </div>
        
        <div class="controls">
            <button onclick="loadImages()">Refresh</button>
        </div>
        
        <div id="imageGrid" class="image-grid"></div>
    </div>

    <script>
        // Load images on page load
        document.addEventListener('DOMContentLoaded', loadImages);
        
        function loadImages() {
            const folder = '{{ folder }}';
            const limit = {{ limit|default(50) }};
            
            fetch(`/browse?folder=${encodeURIComponent(folder)}&limit=${limit}`)
                .then(response => response.json())
                .then(data => {
                    const grid = document.getElementById('imageGrid');
                    grid.innerHTML = '';
                    
                    data.images.forEach(image => {
                        const item = document.createElement('div');
                        item.className = 'image-item';
                        item.innerHTML = `
                            <img src="/image/${encodeURIComponent(image.path)}" 
                                 alt="${image.filename}" 
                                 onclick="window.location.href='/view/${encodeURIComponent(image.path)}'">
                        `;
                        grid.appendChild(item);
                    });
                })
                .catch(error => console.error('Error loading images:', error));
        }
    </script>
</body>
</html>''')

if __name__ == '__main__':
    # Create template files if they don't exist
    create_templates()
    
    # Create test folder if it doesn't exist
    if not os.path.exists(DEFAULT_TEST_FOLDER):
        os.makedirs(DEFAULT_TEST_FOLDER)
        print(f"Created test folder: {DEFAULT_TEST_FOLDER}")
        
    # Run the Flask app
    print(f"Starting web server at http://127.0.0.1:5000/")
    print(f"Default test folder: {DEFAULT_TEST_FOLDER}")
    app.run(debug=True)
