import numpy as np
import random
import cv2
from PIL import Image, ImageOps
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import map_coordinates, gaussian_filter
import math

class MushroomSpatialTransforms:
    """Advanced spatial transforms specifically designed for mushroom classification."""
    
    class RadialDistortion:
        """
        Apply barrel or pincushion distortion to simulate lens effects and natural cap curvature.
        
        Args:
            distortion_type: 'barrel' (outward) or 'pincushion' (inward)
            strength: Distortion strength (0.0-1.0)
            p: Probability of applying transform
        """
        def __init__(self, distortion_type='barrel', strength=0.2, p=0.5):
            self.distortion_type = distortion_type
            self.strength = strength * (1 if distortion_type == 'barrel' else -1)
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array
                img_np = np.array(img).astype(np.float32) / 255.0
                
                h, w = img_np.shape[:2]
                center_x, center_y = w / 2, h / 2
                
                # Create meshgrid for the coordinates
                x = np.arange(w)
                y = np.arange(h)
                X, Y = np.meshgrid(x, y)
                
                # Calculate distance from center and normalize
                dist_x, dist_y = X - center_x, Y - center_y
                r = np.sqrt(dist_x**2 + dist_y**2) / (np.sqrt(center_x**2 + center_y**2))
                
                # Apply distortion
                d = 1.0 + self.strength * (r**2)
                
                # Create new coordinates
                X_distorted = center_x + dist_x * d
                Y_distorted = center_y + dist_y * d
                
                # Clip to valid image coordinates
                X_distorted = np.clip(X_distorted, 0, w - 1).astype(np.float32)
                Y_distorted = np.clip(Y_distorted, 0, h - 1).astype(np.float32)
                
                # Remap image
                result = np.zeros_like(img_np)
                for c in range(img_np.shape[2]):
                    result[:,:,c] = cv2.remap(img_np[:,:,c], X_distorted, Y_distorted, 
                                             interpolation=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_REFLECT)
                
                # Convert back to PIL Image
                return Image.fromarray((result * 255).astype(np.uint8))
            return img

    class ElasticDeformation:
        """
        Apply elastic deformation to simulate natural mushroom growth variations.
        
        Args:
            alpha: Deformation intensity
            sigma: Smoothness of deformation
            p: Probability of applying transform
        """
        def __init__(self, alpha=5.0, sigma=2.0, p=0.5):
            self.alpha = alpha
            self.sigma = sigma
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert PIL Image to numpy array
                img_np = np.array(img)
                
                # Get image shape
                shape = img_np.shape
                
                # Create random displacement fields
                dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), self.sigma) * self.alpha
                dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), self.sigma) * self.alpha
                
                # Create meshgrid for coordinates
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                
                # Displace meshgrid indices
                indices_x = np.reshape(x + dx, (-1, 1))
                indices_y = np.reshape(y + dy, (-1, 1))
                
                # Create distorted image for each channel
                result = np.zeros_like(img_np)
                for c in range(shape[2]):
                    result[:, :, c] = map_coordinates(
                        img_np[:, :, c], 
                        [indices_y.flatten(), indices_x.flatten()],
                        order=1
                    ).reshape(shape[:2])
                    
                return Image.fromarray(result)
            return img

    class CentralFocusZoom:
        """
        Apply selective zoom focused on the center region while maintaining periphery.
        Good for emphasizing cap-stem junction.
        
        Args:
            strength: Zoom strength (0.0-1.0)
            center_size: Relative size of center region (0.0-1.0)
            p: Probability of applying transform
        """
        def __init__(self, strength=0.3, center_size=0.5, p=0.5):
            self.strength = strength
            self.center_size = center_size
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                img_np = np.array(img).astype(np.float32) / 255.0
                h, w = img_np.shape[:2]
                
                # Create meshgrid
                x = np.linspace(-1, 1, w)
                y = np.linspace(-1, 1, h)
                X, Y = np.meshgrid(x, y)
                
                # Calculate radius from center (normalized)
                R = np.sqrt(X**2 + Y**2)
                
                # Create zoom function - more zoom in center, less at edges
                zoom_factor = 1 + self.strength * (1 - np.minimum(1, R / self.center_size))
                
                # Apply non-uniform zoom
                X_zoomed = X / zoom_factor
                Y_zoomed = Y / zoom_factor
                
                # Scale back to image coordinates
                X_zoomed = (X_zoomed + 1) * (w - 1) / 2
                Y_zoomed = (Y_zoomed + 1) * (h - 1) / 2
                
                # Remap image
                result = np.zeros_like(img_np)
                for c in range(img_np.shape[2]):
                    result[:,:,c] = cv2.remap(img_np[:,:,c], 
                                             X_zoomed.astype(np.float32), 
                                             Y_zoomed.astype(np.float32), 
                                             interpolation=cv2.INTER_LINEAR, 
                                             borderMode=cv2.BORDER_REFLECT)
                
                return Image.fromarray((result * 255).astype(np.uint8))
            return img

    class AspectRatioVariation:
        """
        Apply subtle changes to height/width ratio to simulate different growth stages.
        
        Args:
            ratio_range: Range of aspect ratio changes (min, max)
            p: Probability of applying transform
        """
        def __init__(self, ratio_range=(0.9, 1.1), p=0.5):
            self.ratio_range = ratio_range
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Get original size
                width, height = img.size
                
                # Choose random aspect ratio change
                ratio = random.uniform(*self.ratio_range)
                
                # Apply to width or height randomly
                if random.random() < 0.5:
                    new_width = int(width * ratio)
                    new_height = height
                else:
                    new_width = width
                    new_height = int(height * ratio)
                
                # Resize with new aspect ratio
                resized = img.resize((new_width, new_height), Image.BICUBIC)
                
                # Create new image with original size and paste resized image in center
                result = Image.new('RGB', (width, height))
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                result.paste(resized, (paste_x, paste_y))
                
                return result
            return img

    class GridShuffle:
        """
        Divide image into grid cells and randomly shuffle them.
        Forces model to focus on local patterns rather than global structure.
        
        Args:
            grid_size: Number of grid divisions (2=2x2 grid, 3=3x3 grid, etc.)
            p: Probability of applying transform
        """
        def __init__(self, grid_size=2, p=0.3):
            self.grid_size = grid_size
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                width, height = img.size
                cell_width = width // self.grid_size
                cell_height = height // self.grid_size
                
                # Create grid cells
                cells = []
                for y in range(self.grid_size):
                    for x in range(self.grid_size):
                        box = (x * cell_width, y * cell_height, 
                              (x + 1) * cell_width, (y + 1) * cell_height)
                        cell = img.crop(box)
                        cells.append((cell, box))
                
                # Shuffle the cells (but keep the boxes in original order)
                random.shuffle([c for c, _ in cells])
                
                # Create new image and place shuffled cells
                result = img.copy()
                for (cell, box) in cells:
                    result.paste(cell, box)
                
                return result
            return img

    class PolarTransform:
        """
        Convert image to polar coordinates and apply transformations.
        Particularly effective for circular mushroom caps with radial features.
        
        Args:
            rotation: Range of rotation in polar space
            p: Probability of applying transform
        """
        def __init__(self, rotation=(-15, 15), p=0.3):
            self.rotation_range = rotation
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array
                img_np = np.array(img)
                h, w = img_np.shape[:2]
                
                # Find center
                center_x, center_y = w // 2, h // 2
                
                # Determine maximum radius
                max_radius = int(np.sqrt(center_x**2 + center_y**2))
                
                # Create polar image canvas (radius, angle, channels)
                polar_img = np.zeros((max_radius, 360, img_np.shape[2]), dtype=np.uint8)
                
                # Convert to polar coordinates
                for r in range(max_radius):
                    for theta in range(360):
                        # Convert polar to cartesian
                        x = int(center_x + r * np.cos(np.radians(theta)))
                        y = int(center_y + r * np.sin(np.radians(theta)))
                        
                        # Check if within image bounds
                        if 0 <= x < w and 0 <= y < h:
                            polar_img[r, theta] = img_np[y, x]
                
                # Apply transformation in polar space
                # Rotate the polar image (shift in theta direction)
                rotation = random.randint(*self.rotation_range)
                polar_img = np.roll(polar_img, rotation, axis=1)
                
                # Convert back to cartesian
                result = np.zeros_like(img_np)
                for y in range(h):
                    for x in range(w):
                        # Calculate radius and angle
                        dx = x - center_x
                        dy = y - center_y
                        r = int(np.sqrt(dx**2 + dy**2))
                        theta = int(np.degrees(np.arctan2(dy, dx))) % 360
                        
                        # Check bounds
                        if r < max_radius:
                            result[y, x] = polar_img[r, theta]
                
                return Image.fromarray(result)
            return img
            
    class StructuredOcclusion:
        """
        Add realistic occlusions mimicking forest debris or overlapping mushrooms.
        More contextually appropriate than random erasing.
        
        Args:
            num_shapes: Number of occlusion shapes to add
            max_size: Maximum size of occlusions as fraction of image size
            p: Probability of applying transform
        """
        def __init__(self, num_shapes=(1, 3), max_size=0.2, p=0.3):
            self.num_shapes = num_shapes  # Range of number of shapes
            self.max_size = max_size      # Maximum size as fraction of image
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                img_np = np.array(img).copy()
                h, w = img_np.shape[:2]
                
                # Determine number of shapes to add
                n_shapes = random.randint(*self.num_shapes)
                
                # Generate random shapes and add them
                for _ in range(n_shapes):
                    # Randomly select shape type
                    shape_type = random.choice(['leaf', 'branch', 'moss', 'pebble'])
                    
                    # Calculate shape size
                    size = int(min(w, h) * random.uniform(0.05, self.max_size))
                    
                    # Random position
                    x = random.randint(0, w - size//2)
                    y = random.randint(0, h - size//2)
                    
                    # Create mask based on shape type
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if shape_type == 'leaf':
                        # Create leaf-like shape
                        points = []
                        for i in range(6):  # Create a leaf with 6 points
                            angle = i * (2 * np.pi / 6)
                            length = size * (0.5 if i % 2 == 0 else 1.0)
                            px = x + int(length * np.cos(angle))
                            py = y + int(length * np.sin(angle))
                            points.append([px, py])
                        points = np.array([points], dtype=np.int32)
                        cv2.fillPoly(mask, points, 255)
                        
                    elif shape_type == 'branch':
                        # Create elongated shape
                        angle = random.uniform(0, 2 * np.pi)
                        length = size * 3
                        end_x = x + int(length * np.cos(angle))
                        end_y = y + int(length * np.sin(angle))
                        thickness = size // 3
                        cv2.line(mask, (x, y), (end_x, end_y), 255, thickness)
                        
                    elif shape_type == 'moss':
                        # Create irregular blob
                        center = (x, y)
                        axes = (size, size // 2)
                        angle = random.randint(0, 360)
                        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
                        
                    else:  # pebble
                        # Create circular/oval shape
                        center = (x, y)
                        axes = (size // 2, size // 2)
                        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                    
                    # Create occlusion color (earthy tones)
                    if shape_type == 'leaf':
                        color = (random.randint(20, 100), random.randint(50, 150), random.randint(20, 70))
                    elif shape_type == 'branch':
                        color = (random.randint(60, 120), random.randint(30, 80), random.randint(10, 50))
                    elif shape_type == 'moss':
                        color = (random.randint(20, 100), random.randint(80, 180), random.randint(20, 80))
                    else:  # pebble
                        gray = random.randint(70, 200)
                        color = (gray, gray, gray)
                    
                    # Apply occlusion
                    for c in range(3):
                        img_np[:, :, c] = np.where(mask == 255, color[c], img_np[:, :, c])
                
                return Image.fromarray(img_np)
            return img

    class CutMix:
        """
        Cut regions from one image and paste them onto current image.
        
        Args:
            dataset: A dataset that provides other images to mix with
            num_patches: Number of patches to cut and mix
            size_range: Range of relative patch sizes (min, max)
            p: Probability of applying the transform
        """
        def __init__(self, p=0.3, dataset=None, num_patches=(1, 3), size_range=(0.1, 0.3)):
            self.p = p
            self.dataset = dataset
            self.num_patches = num_patches
            self.size_range = size_range
            
        def __call__(self, img):
            if random.random() < self.p and self.dataset is not None:
                # Create a copy of the image
                result = img.copy()
                width, height = result.size
                
                # Choose random number of patches
                n_patches = random.randint(*self.num_patches)
                
                for _ in range(n_patches):
                    # Get random sample from the dataset
                    random_idx = random.randint(0, len(self.dataset) - 1)
                    donor_img, _ = self.dataset[random_idx]
                    
                    # Ensure donor is PIL Image
                    if not isinstance(donor_img, Image.Image):
                        if isinstance(donor_img, torch.Tensor):
                            # Convert tensor to PIL image
                            donor_img = TF.to_pil_image(donor_img)
                        else:
                            # Skip if donor can't be converted
                            continue
                    
                    # Resize donor to same size if needed
                    if donor_img.size != img.size:
                        donor_img = donor_img.resize(img.size, Image.BICUBIC)
                    
                    # Determine patch size
                    patch_size = random.uniform(*self.size_range)
                    patch_w = int(width * patch_size)
                    patch_h = int(height * patch_size)
                    
                    # Random position in donor image
                    donor_x = random.randint(0, width - patch_w)
                    donor_y = random.randint(0, height - patch_h)
                    
                    # Random position in target image
                    target_x = random.randint(0, width - patch_w)
                    target_y = random.randint(0, height - patch_h)
                    
                    # Cut patch from donor
                    patch = donor_img.crop((donor_x, donor_y, donor_x + patch_w, donor_y + patch_h))
                    
                    # Paste patch to target
                    result.paste(patch, (target_x, target_y))
                
                return result
            return img

    class ThinPlateSpline:
        """
        Apply Thin-Plate Spline transformation for smooth non-linear warping.
        Simulates realistic shape deformations while preserving continuity.
        
        Args:
            num_control_points: Number of control points to use
            deformation_strength: Range of deformation
            p: Probability of applying transform
        """
        def __init__(self, num_control_points=5, deformation_strength=0.1, p=0.3):
            self.num_control_points = num_control_points
            self.deformation_strength = deformation_strength
            self.p = p
            
        def __call__(self, img):
            if random.random() < self.p:
                # Convert to numpy array
                img_np = np.array(img).astype(np.uint8)
                h, w = img_np.shape[:2]
                
                # Create source control points in a grid
                n = self.num_control_points
                source_points = np.zeros((n * n, 2), dtype=np.float32)
                step_x = w // (n + 1)
                step_y = h // (n + 1)
                
                # Initialize grid points
                for i in range(n):
                    for j in range(n):
                        source_points[i * n + j, 0] = (j + 1) * step_x
                        source_points[i * n + j, 1] = (i + 1) * step_y
                
                # Create target points with small random displacements
                max_displacement = min(w, h) * self.deformation_strength
                target_points = source_points.copy()
                for i in range(n * n):
                    dx = random.uniform(-max_displacement, max_displacement)
                    dy = random.uniform(-max_displacement, max_displacement)
                    target_points[i, 0] += dx
                    target_points[i, 1] += dy
                
                # Add corner points to maintain image boundaries
                corner_points = np.array([
                    [0, 0],
                    [0, h-1],
                    [w-1, 0],
                    [w-1, h-1]
                ], dtype=np.float32)
                
                source_points = np.vstack([source_points, corner_points])
                target_points = np.vstack([target_points, corner_points])
                
                # Apply Thin-Plate Spline transformation
                tps = cv2.createThinPlateSplineShapeTransformer()
                matches = []
                for i in range(len(source_points)):
                    matches.append(cv2.DMatch(i, i, 0))
                
                source_points = source_points.reshape(1, -1, 2)
                target_points = target_points.reshape(1, -1, 2)
                
                tps.estimateTransformation(target_points, source_points, matches)
                
                # Apply transformation
                result = tps.warpImage(img_np)
                
                return Image.fromarray(result)
            return img

def get_mushroom_spatial_transforms(p=0.5):
    """
    Get a composed transform that applies multiple mushroom-specific spatial transformations.
    Each transform has a probability p of being applied.
    
    Args:
        p: Base probability for each transform (can be overridden individually)
    """
    transforms = [
        MushroomSpatialTransforms.RadialDistortion(p=p),
        MushroomSpatialTransforms.ElasticDeformation(p=p),
        MushroomSpatialTransforms.CentralFocusZoom(p=p),
        MushroomSpatialTransforms.AspectRatioVariation(p=p),
        MushroomSpatialTransforms.StructuredOcclusion(p=p*0.5),  # Lower probability as it's more invasive
        MushroomSpatialTransforms.PolarTransform(p=p*0.3)        # Lower probability as it's more experimental
    ]
    
    # Apply in random order
    random.shuffle(transforms)
    
    return transforms
