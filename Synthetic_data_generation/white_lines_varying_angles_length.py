# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import json

# Configuration
output_dir = r"Path_to_Images\white_lines_varying_angles_sizes"
image_size = (224, 224)
background_noise_levels = [0, 0.1, 0.2, 0.3]  # Noise levels (0 to 1)
min_line_length = 20  # Minimum line length
max_line_length = 100  # Maximum line length
line_color = 'white'  # All lines will be white
line_width = 2  # Width of the lines

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Excel file setup
excel_path = os.path.join(output_dir, "line_labels.xlsx")
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame(columns=[
        'image_id', 'angle_degrees', 'start_point', 'end_point', 
        'noise_level', 'line_length', 'line_width'
    ])

def generate_line(draw, width, height):
    """Generate a white line with random angle and length"""
    # Random line length
    current_length = random.randint(min_line_length, max_line_length)
    
    # Random starting point that allows full line to fit
    x1 = random.randint(current_length, width - current_length - 1)
    y1 = random.randint(current_length, height - current_length - 1)
    
    # Random angle (0-360 degrees)
    angle = random.uniform(0, 2*np.pi)
    
    # Calculate end point
    x2 = x1 + int(current_length * np.cos(angle))
    y2 = y1 + int(current_length * np.sin(angle))
    
    # Draw the line
    draw.line([x1, y1, x2, y2], fill=line_color, width=line_width)
    return angle, (x1, y1), (x2, y2), current_length

def add_noise(image_array, noise_level):
    """Add Gaussian noise to the image"""
    if noise_level == 0:
        return image_array
    noise = np.random.normal(0, noise_level * 255, image_array.shape)
    return np.clip(image_array + noise, 0, 255).astype(np.uint8)

image_counter = 1  # Initialize counter

# Main generation loop
try:
    num_images_to_generate = 100000  # Set your desired number here
    for _ in range(num_images_to_generate):
        # Generate a unique image ID
        image_id = f"Image{image_counter}"
        image_path = os.path.join(output_dir, f"{image_id}.png")
        
        # Create blank image
        image = Image.new('RGB', image_size, (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Generate line with random angle and length
        angle, start_point, end_point, length = generate_line(draw, image_size[0], image_size[1])
        
        # Add random noise to background
        noise_level = random.choice(background_noise_levels)
        image_array = np.array(image)
        noisy_image_array = add_noise(image_array, noise_level)
        image = Image.fromarray(noisy_image_array)
        
        # Save the image
        image.save(image_path)
        
        # Add to DataFrame
        new_row = {
            'image_id': image_id,
            'angle_degrees': round(np.degrees(angle), 2),  # Convert to degrees with 2 decimal places
            'start_point': str(start_point),
            'end_point': str(end_point),
            'noise_level': noise_level,
            'line_length': length,
            'line_width': line_width
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel after each image
        df.to_excel(excel_path, index=False)
        
        # Changed from degree symbol to 'deg' to avoid encoding issues
        print(f"Generated {image_id} - Angle: {round(np.degrees(angle), 2)} deg, Length: {length} px")

        image_counter += 1

except KeyboardInterrupt:
    print("Dataset generation stopped by user")

# Final save before exiting
df.to_excel(excel_path, index=False)
print(f"Data saved to {excel_path}")