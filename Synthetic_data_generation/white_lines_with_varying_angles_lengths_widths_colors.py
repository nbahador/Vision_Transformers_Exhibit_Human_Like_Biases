# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json
from collections import OrderedDict

# Configuration
output_dir = r"Path_to_Images\white_lines_with_varying_angles_lengths_widths_colors"
image_size = (224, 224)
background_noise_levels = [0, 0.1, 0.2, 0.3]  # Noise levels (0 to 1)

# Line properties ranges
line_properties = {
    'length': (20, 100),    # min, max length in pixels
    'width': (1, 5),        # min, max width in pixels
    'colors': [             # Available line colors
        'red', 'green', 'blue', 'yellow', 
        'magenta', 'cyan', 'white', 'orange',
        'purple', 'pink', 'teal'
    ]
}

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Excel file setup
excel_path = os.path.join(output_dir, "line_properties.xlsx")
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame(columns=[
        'image_id', 'angle_degrees', 'start_point', 'end_point',
        'length', 'width', 'color_rgb', 'color_name',
        'noise_level'
    ])

def generate_line(draw, width, height):
    """Generate a line with random properties"""
    # Randomly select properties
    length = random.randint(*line_properties['length'])
    line_width = random.randint(*line_properties['width'])
    color_name = random.choice(line_properties['colors'])
    
    # Get RGB values for the color
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'magenta': (255, 0, 255),
        'cyan': (0, 255, 255),
        'white': (255, 255, 255),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'teal': (0, 128, 128)
    }
    color_rgb = color_map[color_name]
    
    # Calculate maximum possible starting position
    buffer = length + line_width
    x1 = random.randint(buffer, width - buffer - 1)
    y1 = random.randint(buffer, height - buffer - 1)
    
    # Random angle (0-360 degrees)
    angle = random.uniform(0, 2*np.pi)
    
    # Calculate end point
    x2 = x1 + int(length * np.cos(angle))
    y2 = y1 + int(length * np.sin(angle))
    
    # Draw the line
    draw.line([x1, y1, x2, y2], fill=color_rgb, width=line_width)
    
    return {
        'angle': angle,
        'start': (x1, y1),
        'end': (x2, y2),
        'length': length,
        'width': line_width,
        'color_rgb': color_rgb,
        'color_name': color_name
    }

def add_noise(image_array, noise_level):
    """Add Gaussian noise to the image"""
    if noise_level == 0:
        return image_array
    noise = np.random.normal(0, noise_level * 255, image_array.shape)
    return np.clip(image_array + noise, 0, 255).astype(np.uint8)

image_counter = 1

# Main generation loop
try:
    num_images_to_generate = 100000  # Number of images to create
    for _ in range(num_images_to_generate):
        # Create blank image
        image = Image.new('RGB', image_size, (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Generate line with random properties
        line_data = generate_line(draw, image_size[0], image_size[1])
        
        # Add background noise
        noise_level = random.choice(background_noise_levels)
        noisy_image = add_noise(np.array(image), noise_level)
        image = Image.fromarray(noisy_image)
        
        # Save image
        image_id = f"line_{image_counter:04d}"
        image_path = os.path.join(output_dir, f"{image_id}.png")
        image.save(image_path)
        
        # Add to DataFrame
        new_row = OrderedDict([
            ('image_id', image_id),
            ('angle_degrees', round(np.degrees(line_data['angle']), 2)),
            ('start_point', str(line_data['start'])),
            ('end_point', str(line_data['end'])),
            ('length', line_data['length']),
            ('width', line_data['width']),
            ('color_rgb', str(line_data['color_rgb'])),
            ('color_name', line_data['color_name']),
            ('noise_level', noise_level)
        ])
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Periodically save progress
        if image_counter % 10 == 0:
            df.to_excel(excel_path, index=False)
            print(f"Saved {image_counter} images...")
        
        image_counter += 1

except KeyboardInterrupt:
    print("\nGeneration stopped by user")

# Final save
df.to_excel(excel_path, index=False)
print(f"\nCompleted {image_counter-1} images")
print(f"Data saved to {excel_path}")