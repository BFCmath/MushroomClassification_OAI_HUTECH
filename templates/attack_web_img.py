# Web-based image viewer for test folder images

from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import json
from pathlib import Path
# import numpy as np # <-- Not used, can be removed
from PIL import Image, UnidentifiedImageError # <-- Import specific error
import io
import base64

# Constants
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
# Use absolute path joining for reliability
DEFAULT_TEST_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'attack_dataset/colorize_mushroom_test')) # <-- Safer default path

app = Flask(__name__)
# Ensure templates folder is correctly identified relative to this script
app.template_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))

def get_images_from_folder(folder_path):
    """Get all image files from a folder."""
    # Ensure folder_path is absolute and normalized
    folder_path = os.path.abspath(folder_path)
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"Warning: Folder not found or is not a directory: {folder_path}")
        return [] # Return empty list if folder doesn't exist

    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        # Case-insensitive globbing might be better depending on OS/needs
        # Using separate upper/lower for cross-platform safety
        image_paths.extend(list(folder.glob(f"*{ext}")))
        image_paths.extend(list(folder.glob(f"*{ext.upper()}")))

    # Remove duplicates that might arise from case differences if filesystem is case-insensitive
    # Convert to string first for reliable comparison
    image_paths = sorted(list(set(str(p) for p in image_paths)))
    # Convert back to Path objects if needed elsewhere, but strings are fine for this app
    # image_paths = [Path(p) for p in image_paths] # Keep as strings for simplicity

    return image_paths # Return list of string paths

def get_image_metadata(image_path_str):
    """Get basic metadata for an image."""
    metadata = {
        'filename': os.path.basename(image_path_str),
        'path': image_path_str, # Store the string path directly
        'error': None,
        'width': None,
        'height': None,
        'format': None,
        'mode': None
    }
    try:
        # Use Path object internally for opening
        image_path = Path(image_path_str)
        if not image_path.is_file():
             raise FileNotFoundError("Image file not found")

        with Image.open(image_path) as img:
            metadata.update({
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            })
    except UnidentifiedImageError:
         # Handle files that are not valid images gracefully
         error_msg = "Cannot identify image file (possibly corrupted or not an image)"
         print(f"Error getting metadata for {image_path_str}: {error_msg}")
         metadata['error'] = error_msg
    except FileNotFoundError as e:
        print(f"Error getting metadata for {image_path_str}: {e}")
        metadata['error'] = str(e)
    except Exception as e:
        # Catch other potential PIL errors
        print(f"Unexpected error getting metadata for {image_path_str}: {e}")
        metadata['error'] = f"Unexpected error: {str(e)}"
    return metadata

def image_to_base64(image_path_str, max_width=250): # Smaller default for grid thumbnails
    """Convert image to base64 for embedding in HTML (used for single view)."""
    try:
        image_path = Path(image_path_str)
        if not image_path.is_file():
             raise FileNotFoundError("Image file not found")

        with Image.open(image_path) as img:
            # Ensure format is determined, default to JPEG if missing
            img_format = img.format if img.format else 'JPEG'
            # Handle formats that might not support direct saving (like WEBP sometimes)
            if img.mode == 'P' and img_format not in ['PNG', 'GIF']: # Palette mode often needs conversion
                 img = img.convert('RGB')
            elif img.mode == 'RGBA' and img_format == 'JPEG': # JPEG doesn't support alpha
                 img = img.convert('RGB')


            # Resize if needed
            w, h = img.size
            if w > max_width:
                ratio = max_width / w
                new_size = (int(w * ratio), int(h * ratio))
                # Use ANTIALIAS for better quality resizing (LANCZOS is good too)
                img = img.resize(new_size, Image.Resampling.LANCZOS) # Updated resizing constant

            # Convert to base64
            buffer = io.BytesIO()
            # Use the determined format, handle potential errors during save
            try:
                img.save(buffer, format=img_format)
            except OSError as save_err:
                 # Try saving as JPEG as a fallback if original format failed
                 print(f"Warning: Could not save in original format {img_format} for {image_path_str}. Trying JPEG. Error: {save_err}")
                 img = img.convert('RGB') # Ensure RGB for JPEG
                 img_format = 'JPEG'
                 buffer = io.BytesIO() # Reset buffer
                 img.save(buffer, format=img_format)

            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/{img_format.lower()};base64,{img_str}"
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"Error converting image {image_path_str} to base64: {e}")
        return None # Return None if file not found or not an image
    except Exception as e:
        print(f"Unexpected error converting image {image_path_str} to base64: {e}")
        return None # Return None for other errors


@app.route('/')
def index():
    """Home page - display the main image browser interface."""
    image_folder = request.args.get('folder', DEFAULT_TEST_FOLDER)

    # Create test folder if it doesn't exist (only if using the default)
    if image_folder == DEFAULT_TEST_FOLDER and not os.path.exists(DEFAULT_TEST_FOLDER):
        try:
            os.makedirs(DEFAULT_TEST_FOLDER)
            print(f"Created default test folder: {DEFAULT_TEST_FOLDER}")
        except OSError as e:
            print(f"Error creating default test folder {DEFAULT_TEST_FOLDER}: {e}")
            # Optionally, you might want to redirect or show an error page here

    # --- FIX: Render the correct template file ---
    return render_template('test_index.html',
                          folder=image_folder,
                          default_folder=DEFAULT_TEST_FOLDER)

@app.route('/browse')
def browse():
    """API endpoint to get images in a folder."""
    folder = request.args.get('folder', DEFAULT_TEST_FOLDER)
    limit = request.args.get('limit', None)
    # No need for grid_view param here, it's handled client-side now

    if limit:
        try:
            limit = int(limit)
            if limit <= 0: # Handle non-positive limit
                limit = None
        except ValueError:
            limit = None # Ignore invalid limit values

    # Get image paths (list of strings)
    image_paths = get_images_from_folder(folder)
    total_found = len(image_paths) # Count before limiting

    # Apply limit if specified
    if limit and limit < len(image_paths):
        image_paths = image_paths[:limit]

    # Get metadata for each image
    images_metadata = []
    for path_str in image_paths:
        metadata = get_image_metadata(path_str)
        # Only include if metadata could be partially retrieved (has filename/path)
        if metadata.get('filename'):
             images_metadata.append(metadata)

    return jsonify({
        'folder': folder,
        'limit': limit,
        'total_found': total_found, # Inform client about total available
        'count_returned': len(images_metadata), # How many are actually in the list
        'images': images_metadata # This list contains only JSON-serializable dicts
    })

@app.route('/image/<path:image_path_str>')
def serve_image(image_path_str):
    """Serve an image file directly.
       The path received here should be the absolute path.
    """
    # Security check: Ensure the requested path is within expected parent directories
    # This is a basic check; more robust validation might be needed depending on use case.
    safe_base_dirs = [DEFAULT_TEST_FOLDER, os.path.abspath(os.path.dirname(DEFAULT_TEST_FOLDER))] # Allow default and its parent
    abs_image_path = os.path.abspath(image_path_str)

    is_safe = False
    for safe_dir in safe_base_dirs:
        if os.path.commonpath([abs_image_path, safe_dir]) == safe_dir:
            is_safe = True
            break

    if not is_safe or '..' in image_path_str: # Basic directory traversal check
        print(f"Warning: Denying access to potentially unsafe path: {image_path_str}")
        return "Access denied", 403 # Forbidden

    try:
        directory = os.path.dirname(abs_image_path)
        basename = os.path.basename(abs_image_path)
        # Check if file exists before sending
        if not os.path.isfile(os.path.join(directory, basename)):
             return "Image not found", 404
        return send_from_directory(directory, basename)
    except Exception as e:
        print(f"Error serving image {image_path_str}: {e}")
        return "Error serving image", 500


@app.route('/view/<path:image_path_str>')
def view_image(image_path_str):
    """View a single image with details."""
    # Normalize path received from URL
    image_path_abs = os.path.abspath(image_path_str)

    # Basic safety check (similar to serve_image)
    safe_base_dirs = [DEFAULT_TEST_FOLDER, os.path.abspath(os.path.dirname(DEFAULT_TEST_FOLDER))]
    is_safe = False
    for safe_dir in safe_base_dirs:
        if os.path.commonpath([image_path_abs, safe_dir]) == safe_dir:
            is_safe = True
            break

    if not is_safe or '..' in image_path_str:
        print(f"Warning: Denying view access to potentially unsafe path: {image_path_str}")
        return "Access denied", 403

    # Get metadata
    metadata = get_image_metadata(image_path_abs)

    # If metadata retrieval failed significantly (e.g., file not found)
    if metadata.get('error') and not metadata.get('width'):
        # Maybe render an error page or redirect
        return f"Error loading image details: {metadata['error']}", 404

    # Get base64 representation (for display in the viewer page)
    # Increase max_width for the single view page
    image_data = image_to_base64(image_path_abs, max_width=1200)

    return render_template('image_view.html',
                          image=metadata,       # Pass the metadata dictionary
                          image_data=image_data) # Pass the base64 string or None


# --- Removed the '/grid' route as it's now handled by the main '/' page and JavaScript ---

# Create templates directory and files if they don't exist
def create_templates():
    # Ensure templates directory is relative to the script file
    script_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(script_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    print(f"Ensuring templates directory exists: {templates_dir}")

    # --- Create test_index.html (using UTF-8 consistently) ---
    index_html_path = os.path.join(templates_dir, 'test_index.html')
    if not os.path.exists(index_html_path):
        print(f"Creating template: {index_html_path}")
        with open(index_html_path, 'w', encoding='utf-8') as f:
            # (Content is the same as provided, just ensure variable names match JS)
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Image Viewer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #ddd; }
        .header h1 { margin: 0 0 5px 0; }
        .controls { margin-bottom: 20px; padding: 15px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; flex-wrap: wrap; gap: 15px; align-items: center; }
        .controls > div { display: flex; align-items: center; gap: 8px; }
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
        .image-card { background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 10px; transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; overflow: hidden; display: flex; flex-direction: column; height: 100%; }
        .image-card:hover { transform: translateY(-3px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }
        .image-card a { display: block; margin-bottom: 10px; }
        .image-card img { width: 100%; height: 150px; object-fit: cover; border-radius: 5px; display: block; background-color: #eee; } /* Added background color for loading state */
        .image-card .error-thumb { width: 100%; height: 150px; display: flex; align-items: center; justify-content: center; background-color: #eee; color: #a00; font-size: 12px; text-align: center; border-radius: 5px; }
        .image-info { margin-top: auto; font-size: 13px; line-height: 1.4; word-break: break-all; } /* Ensure info stays at bottom */
        .image-info strong { font-size: 14px; }
        .loader { text-align: center; padding: 40px; font-size: 1.2em; color: #555; }
        .view-toggle { margin-bottom: 15px; }
        .btn { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; transition: background-color 0.2s; }
        .btn:hover { background: #0056b3; }
        .btn-secondary { background-color: #6c757d; }
        .btn-secondary:hover { background-color: #5a6268; }
        input[type="text"], input[type="number"] { padding: 8px 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 14px; }
        input[type="text"] { flex-grow: 1; min-width: 250px; }
        input[type="number"] { width: 80px; }
        .error { color: #d9534f; padding: 10px 15px; background: #f2dede; border: 1px solid #ebccd1; border-radius: 5px; margin-bottom: 15px; }
        #imageList table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); overflow: hidden; }
        #imageList th, #imageList td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        #imageList th { background-color: #f8f9fa; font-weight: 600; }
        #imageList tr:last-child td { border-bottom: none; }
        #imageList img { width: 80px; height: 60px; object-fit: cover; border-radius: 4px; vertical-align: middle; }
        #imageList .error-thumb { width: 80px; height: 60px; display: flex; align-items: center; justify-content: center; background-color: #eee; color: #a00; font-size: 11px; text-align: center; border-radius: 4px; vertical-align: middle;}

    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Image Viewer</h1>
            <p>Browsing folder: <code id="currentFolderPath">{{ folder }}</code></p>
        </div>

        <div class="controls">
            <div>
                <label for="folderInput">Folder path:</label>
                <input type="text" id="folderInput" value="{{ folder }}">
                <button class="btn" onclick="loadImages()">Browse</button>
            </div>
            <div>
                <label for="limitInput">Limit:</label>
                <input type="number" id="limitInput" value="50" min="1">
            </div>
            <!-- View toggle is now a button -->
        </div>

        <div class="view-toggle">
            <button class="btn btn-secondary" id="viewModeBtn" onclick="toggleViewMode()">Switch to List View</button>
            <span id="imageCountInfo" style="margin-left: 15px; color: #555;"></span>
        </div>

        <div id="error" class="error" style="display: none;"></div>
        <div id="loader" class="loader" style="display: none;">Loading images...</div>
        <div id="imageGrid" class="image-grid"></div>
        <div id="imageList" style="display: none;"></div> <!-- Initially hidden -->
    </div>

    <script>
        let currentViewMode = 'grid'; // Start with grid view
        const defaultFolder = "{{ default_folder }}"; // Get default folder from Flask

        // Function to safely create image source URL
        function getImageUrl(imagePath) {
            // Ensure the path is properly encoded for the URL
            // Use encodeURIComponent on the full path passed to the route
            return `/image/${encodeURIComponent(imagePath)}`;
        }

        // Function to safely create view URL
        function getViewUrl(imagePath) {
             return `/view/${encodeURIComponent(imagePath)}`;
        }

        document.addEventListener('DOMContentLoaded', function() {
            // Set initial folder input value (might differ from default if query param is used)
            document.getElementById('folderInput').value = document.getElementById('currentFolderPath').textContent;
            // Load images on initial page load
            loadImages();
        });

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
                gridView.style.display = 'grid'; // Changed from 'block' to 'grid'
                listView.style.display = 'none';
                viewBtn.textContent = 'Switch to List View';
                currentViewMode = 'grid';
            }
        }

        function loadImages() {
            const folderInput = document.getElementById('folderInput');
            const limitInput = document.getElementById('limitInput');
            const folder = folderInput.value.trim() || defaultFolder; // Use default if empty
            const limit = limitInput.value;

            // Update displayed folder path
            document.getElementById('currentFolderPath').textContent = folder;
            // Update input field in case default was used
            folderInput.value = folder;

            const loader = document.getElementById('loader');
            const errorDiv = document.getElementById('error');
            const imageGrid = document.getElementById('imageGrid');
            const imageList = document.getElementById('imageList');
            const imageCountInfo = document.getElementById('imageCountInfo');

            loader.style.display = 'block';
            errorDiv.style.display = 'none'; // Hide previous errors
            imageGrid.innerHTML = ''; // Clear previous grid
            imageList.innerHTML = ''; // Clear previous list
            imageCountInfo.textContent = ''; // Clear count info

            // Construct fetch URL
            let fetchUrl = `/browse?folder=${encodeURIComponent(folder)}`;
            if (limit) {
                fetchUrl += `&limit=${limit}`;
            }

            fetch(fetchUrl)
                .then(response => {
                    if (!response.ok) {
                        // Handle HTTP errors like 404, 500
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    loader.style.display = 'none';

                    if (data.images && data.images.length > 0) {
                        imageCountInfo.textContent = `Showing ${data.count_returned} of ${data.total_found} images found.`;
                        populateViews(data.images);
                    } else if (data.total_found === 0) {
                         imageCountInfo.textContent = 'No images found in this folder.';
                         showError(`No supported image files found in the specified folder: ${data.folder}`);
                    } else {
                        // Handle case where limit might be 0 or API returned empty list unexpectedly
                        imageCountInfo.textContent = `Found ${data.total_found} images, but none are displayed (check limit or potential errors).`;
                        console.warn("Received data but image list is empty:", data);
                    }

                     // Set view based on current mode (in case it was toggled before reload)
                     if(currentViewMode === 'list') {
                         document.getElementById('imageGrid').style.display = 'none';
                         document.getElementById('imageList').style.display = 'block';
                     } else {
                         document.getElementById('imageGrid').style.display = 'grid';
                         document.getElementById('imageList').style.display = 'none';
                     }

                })
                .catch(error => {
                    loader.style.display = 'none';
                    showError(`Error loading images: ${error.message}. Check if the folder path is correct and the server is running.`);
                    console.error('Fetch error:', error);
                });
        }

        function populateViews(images) {
            const gridContainer = document.getElementById('imageGrid');
            const listContainer = document.getElementById('imageList');
            let listTableHTML = `
                <table>
                    <thead>
                        <tr style="background: #f5f5f5;">
                            <th style="text-align: left; padding: 8px;">Preview</th>
                            <th style="text-align: left; padding: 8px;">Filename</th>
                            <th style="text-align: left; padding: 8px;">Dimensions</th>
                            <th style="text-align: left; padding: 8px;">Format</th>
                            <th style="text-align: left; padding: 8px;">Details</th>
                        </tr>
                    </thead>
                <tbody>`;

            images.forEach(image => {
                const imageUrl = image.path ? getImageUrl(image.path) : ''; // Get safe URL
                const viewUrl = image.path ? getViewUrl(image.path) : '#';
                const dimensions = image.width && image.height ? `${image.width} × ${image.height} px` : 'N/A';
                const format = image.format || 'N/A';
                const filename = image.filename || 'Unknown Filename';

                // --- Grid Card ---
                const card = document.createElement('div');
                card.className = 'image-card';
                let imgElementHTML;
                if (image.error) {
                    imgElementHTML = `<div class="error-thumb" title="${image.error}">⚠️ Error<br/>(${filename})</div>`;
                } else if (imageUrl) {
                    imgElementHTML = `<img src="${imageUrl}" alt="${filename}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'error-thumb\\'>⚠️ Invalid Image</div>';">`;
                } else {
                    imgElementHTML = `<div class="error-thumb">Missing Path</div>`;
                }

                card.innerHTML = `
                    <a href="${viewUrl}" target="_blank">
                       ${imgElementHTML}
                    </a>
                    <div class="image-info">
                        <div><strong>${filename}</strong></div>
                        <div>${dimensions}</div>
                        ${image.error ? `<div style="color: red; font-size: 11px;">Error: ${image.error}</div>` : ''}
                    </div>
                `;
                gridContainer.appendChild(card);

                // --- List Row ---
                 let listImgElementHTML;
                 if (image.error) {
                     listImgElementHTML = `<div class="error-thumb" title="${image.error}">⚠️ Error</div>`;
                 } else if (imageUrl) {
                     listImgElementHTML = `<img src="${imageUrl}" alt="${filename}" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'error-thumb\\'>⚠️ Error</div>';">`;
                 } else {
                      listImgElementHTML = `<div class="error-thumb">Missing Path</div>`;
                 }

                listTableHTML += `
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 8px;">
                            <a href="${viewUrl}" target="_blank">
                               ${listImgElementHTML}
                            </a>
                        </td>
                        <td style="padding: 8px; word-break: break-all;">${filename}</td>
                        <td style="padding: 8px;">${dimensions}</td>
                        <td style="padding: 8px;">${format}</td>
                        <td style="padding: 8px;"><a href="${viewUrl}" target="_blank">View Details</a></td>
                    </tr>
                `;
            });

            listTableHTML += `</tbody></table>`;
            listContainer.innerHTML = listTableHTML;
        }


        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
    </script>
</body>
</html>''')
    else:
        print(f"Template already exists: {index_html_path}")

    # --- Create image_view.html (using UTF-8 consistently) ---
    image_view_html_path = os.path.join(templates_dir, 'image_view.html')
    if not os.path.exists(image_view_html_path):
        print(f"Creating template: {image_view_html_path}")
        with open(image_view_html_path, 'w', encoding='utf-8') as f:
             # (Content is the same as provided, ensure variables like image.width exist or are handled)
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>{{ image.filename if image else 'Image' }} - Image Viewer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }
        .container { max-width: 1200px; margin: 20px auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        .header { margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #eee; }
        .header h1 { margin: 0; word-break: break-all; }
        .image-container { text-align: center; margin-bottom: 30px; background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .image-container img { max-width: 100%; max-height: 80vh; display: block; margin: 0 auto; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .image-container .error { color: #d9534f; font-weight: bold; padding: 30px;}
        .metadata { background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #eee; }
        .metadata h2 { margin-top: 0; margin-bottom: 15px; font-size: 1.3em; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 8px;}
        .metadata table { width: 100%; border-collapse: collapse; }
        .metadata td, .metadata th { padding: 10px 0; text-align: left; border-bottom: 1px solid #eee; vertical-align: top; }
        .metadata tr:last-child td, .metadata tr:last-child th { border-bottom: none; }
        .metadata th { width: 140px; font-weight: 600; color: #555; padding-right: 15px;}
        .metadata td { word-break: break-all; } /* Handle long paths */
        .back-button { margin-bottom: 20px; }
        .btn { padding: 10px 18px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 14px; transition: background-color 0.2s; }
        .btn:hover { background: #0056b3; }
        .error-message { color: #d9534f; font-weight: bold; background-color: #f2dede; padding: 10px; border-radius: 5px; border: 1px solid #ebccd1; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="back-button">
                <!-- Use JavaScript back if referrer is from the same origin, otherwise link to home -->
                <a href="#" onclick="goBack(); return false;" class="btn">← Back</a>
            </div>
            <h1>{{ image.filename if image else 'Error Loading Image' }}</h1>
            {% if image.error %}
                <p class="error-message">Error: {{ image.error }}</p>
            {% endif %}
        </div>

        <div class="image-container">
            {% if image_data %}
                <img src="{{ image_data }}" alt="{{ image.filename if image else 'Image' }}">
            {% elif image and not image.error %}
                 <!-- If no base64, maybe try linking directly? Be careful with large images -->
                 <p>Could not generate preview. <a href="{{ getImageUrl(image.path) }}" target="_blank">View original image</a>.</p>
            {% else %}
                <p class="error">Image preview could not be loaded.</p>
            {% endif %}
        </div>

        {% if image %}
        <div class="metadata">
            <h2>Image Metadata</h2>
            <table>
                <tr>
                    <th>Filename</th>
                    <td>{{ image.filename | default('N/A') }}</td>
                </tr>
                <tr>
                    <th>Path</th>
                    <td>{{ image.path | default('N/A') }}</td>
                </tr>
                <tr>
                    <th>Dimensions</th>
                    {# Handle cases where width/height might be None #}
                    <td>{{ '%s × %s px' % (image.width | default('?'), image.height | default('?')) if image.width or image.height else 'N/A' }}</td>
                </tr>
                <tr>
                    <th>Format</th>
                    <td>{{ image.format | default('N/A') }}</td>
                </tr>
                <tr>
                    <th>Mode</th>
                    <td>{{ image.mode | default('N/A') }}</td>
                </tr>
                {% if image.error %}
                <tr>
                    <th>Loading Error</th>
                    <td class="error-message">{{ image.error }}</td>
                </tr>
                {% endif %}
            </table>
        </div>
        {% else %}
        <div class="metadata">
             <h2>Image Metadata</h2>
             <p>Could not load image metadata.</p>
        </div>
        {% endif %}
    </div>
    <script>
        function goBack() {
            // If the previous page was on the same site, go back, otherwise go to home.
            if (document.referrer && document.referrer.startsWith(window.location.origin)) {
                history.back();
            } else {
                window.location.href = '/'; // Go to home page
            }
        }
    </script>
</body>
</html>''')
    else:
        print(f"Template already exists: {image_view_html_path}")

    # --- Remove creation for grid.html as it's not used by a route anymore ---
    # grid_html_path = os.path.join(templates_dir, 'grid.html')
    # if os.path.exists(grid_html_path):
    #     try:
    #         os.remove(grid_html_path)
    #         print(f"Removed unused template: {grid_html_path}")
    #     except OSError as e:
    #         print(f"Could not remove unused template {grid_html_path}: {e}")


if __name__ == '__main__':
    # Create template files first
    create_templates()

    # Create default test folder if it doesn't exist
    if not os.path.exists(DEFAULT_TEST_FOLDER):
        try:
            os.makedirs(DEFAULT_TEST_FOLDER)
            print(f"Created default test folder: {DEFAULT_TEST_FOLDER}")
            # Optional: Add a placeholder file so the folder isn't empty
            placeholder_file = os.path.join(DEFAULT_TEST_FOLDER, 'put_images_here.txt')
            if not os.path.exists(placeholder_file):
                with open(placeholder_file, 'w') as f:
                    f.write("Place your test images (.jpg, .png, etc.) in this folder.")
        except OSError as e:
            print(f"Error creating default test folder {DEFAULT_TEST_FOLDER}: {e}")
            # Consider exiting if the default folder cannot be created

    # Run the Flask app
    print("-" * 40)
    print(f"Starting Flask Image Viewer")
    print(f"Default test folder: {DEFAULT_TEST_FOLDER}")
    print(f"Templates folder: {app.template_folder}")
    print(f"Access the viewer at: http://127.0.0.1:5000/")
    print("-" * 40)
    # Turn off reloader if template creation happens inside main,
    # or ensure create_templates() is idempotent. debug=True enables the reloader.
    app.run(debug=True) # Use debug=False for production