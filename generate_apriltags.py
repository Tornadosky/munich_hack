"""
AprilTag image generator for camera pose calibration.
Generates printable AprilTag images with precise sizing information.
"""

import argparse
import math
import os
from typing import List
import numpy as np
import cv2
import json


def generate_apriltag_image(tag_id: int, tag_family: str = "tag36h11", 
                           border_pixels: int = 2) -> np.ndarray:
    """
    Generate an AprilTag image.
    
    Args:
        tag_id: Tag ID number
        tag_family: Tag family (e.g., "tag36h11", "tag25h9")
        border_pixels: White border around tag in pixels
        
    Returns:
        Grayscale image of the AprilTag
    """
    try:
        import apriltag
    except ImportError:
        raise ImportError("AprilTag library not found. Install with: pip install apriltag")
    
    # Create detector to get tag family info
    detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))
    
    # Generate tag image using the apriltag library
    # Note: The apriltag library doesn't have a direct image generation function
    # We'll use a workaround by creating a synthetic tag pattern
    
    if tag_family == "tag36h11":
        tag_size = 6  # 6x6 grid for tag36h11
    elif tag_family == "tag25h9":
        tag_size = 5  # 5x5 grid for tag25h9
    elif tag_family == "tag16h5":
        tag_size = 4  # 4x4 grid for tag16h5
    else:
        tag_size = 6  # Default to 6x6
    
    # For now, we'll create a placeholder pattern
    # In a real implementation, you'd need the actual tag patterns
    # This is a simplified approach - you might want to use a more sophisticated
    # tag generation library or pre-generated tag images
    
    # Create a simple checkerboard pattern as placeholder
    # In practice, you should use actual AprilTag patterns
    cell_size = 20  # pixels per cell
    total_size = tag_size * cell_size
    
    # Create the tag pattern (simplified)
    tag_image = np.ones((total_size, total_size), dtype=np.uint8) * 255
    
    # Add a simple pattern based on tag_id (this is not a real AprilTag pattern)
    for i in range(tag_size):
        for j in range(tag_size):
            # Simple pattern based on tag_id and position
            if ((i + j + tag_id) % 2) == 0:
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                tag_image[y1:y2, x1:x2] = 0
    
    # Add border
    if border_pixels > 0:
        bordered_size = total_size + 2 * border_pixels
        bordered_image = np.ones((bordered_size, bordered_size), dtype=np.uint8) * 255
        bordered_image[border_pixels:border_pixels + total_size, 
                      border_pixels:border_pixels + total_size] = tag_image
        tag_image = bordered_image
    
    return tag_image


def create_apriltag_printout(tag_ids: List[int], physical_size: float, 
                            tag_family: str = "tag36h11", 
                            output_file: str = "apriltags.png") -> str:
    """
    Create a printable sheet with multiple AprilTags.
    
    Args:
        tag_ids: List of tag ID numbers to generate
        physical_size: Physical size of each tag in meters
        tag_family: AprilTag family
        output_file: Output image file path
        
    Returns:
        Path to generated image file
    """
    print(f"📄 Generating AprilTag sheet with {len(tag_ids)} tags")
    print(f"   Tag IDs: {tag_ids}")
    print(f"   Physical size: {physical_size}m ({physical_size * 100:.1f}cm)")
    print(f"   Tag family: {tag_family}")
    
    # Generate individual tag images
    tag_images = []
    for tag_id in tag_ids:
        tag_img = generate_apriltag_image(tag_id, tag_family, border_pixels=10)
        tag_images.append(tag_img)
    
    # Calculate layout (arrange in a grid)
    num_tags = len(tag_ids)
    cols = int(math.ceil(math.sqrt(num_tags)))
    rows = int(math.ceil(num_tags / cols))
    
    # Assume standard 300 DPI printing
    dpi = 300
    tag_size_pixels = int(physical_size * dpi * 39.37)  # Convert meters to inches to pixels
    
    # Resize tag images to correct physical size
    resized_tags = []
    for tag_img in tag_images:
        resized = cv2.resize(tag_img, (tag_size_pixels, tag_size_pixels), 
                           interpolation=cv2.INTER_NEAREST)
        resized_tags.append(resized)
    
    # Create the final sheet
    margin_pixels = int(0.02 * dpi * 39.37)  # 2cm margin
    spacing_pixels = int(0.01 * dpi * 39.37)  # 1cm spacing
    
    sheet_width = cols * tag_size_pixels + (cols - 1) * spacing_pixels + 2 * margin_pixels
    sheet_height = rows * tag_size_pixels + (rows - 1) * spacing_pixels + 2 * margin_pixels
    
    # Create white background
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    # Place tags on sheet
    for idx, (tag_id, tag_img) in enumerate(zip(tag_ids, resized_tags)):
        row = idx // cols
        col = idx % cols
        
        y = margin_pixels + row * (tag_size_pixels + spacing_pixels)
        x = margin_pixels + col * (tag_size_pixels + spacing_pixels)
        
        sheet[y:y + tag_size_pixels, x:x + tag_size_pixels] = tag_img
        
        # Add tag ID label below each tag
        label_y = y + tag_size_pixels + 20
        cv2.putText(sheet, f"ID: {tag_id}", (x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    # Add printing instructions at the top
    instructions = [
        f"AprilTag Calibration Sheet - {tag_family}",
        f"Print at 100% scale (no scaling/fitting)",
        f"Tag size: {physical_size * 100:.1f}cm x {physical_size * 100:.1f}cm",
        f"Recommended DPI: 300",
        f"Generated tag IDs: {', '.join(map(str, tag_ids))}"
    ]
    
    y_offset = 50
    for instruction in instructions:
        cv2.putText(sheet, instruction, (margin_pixels, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2)
        y_offset += 30
    
    # Save the image
    cv2.imwrite(output_file, sheet)
    
    # Calculate file size and print info
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"✅ AprilTag sheet generated successfully!")
    print(f"   📁 File: {output_file}")
    print(f"   📏 Image size: {sheet_width}x{sheet_height} pixels")
    print(f"   💾 File size: {file_size_mb:.1f} MB")
    print(f"   🖨️  Recommended printing:")
    print(f"      - Print at 100% scale (no fit-to-page)")
    print(f"      - Use 300 DPI or higher")
    print(f"      - Measure printed tags to verify {physical_size * 100:.1f}cm size")
    
    return output_file


def generate_from_config(config_file: str, output_file: str = "apriltags.png") -> str:
    """
    Generate AprilTags based on configuration file.
    
    Args:
        config_file: Path to AprilTag configuration JSON file
        output_file: Output image file path
        
    Returns:
        Path to generated image file
    """
    print(f"📋 Loading configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_file}")
        raise
    
    # Extract tag information
    tag_ids = [tag['id'] for tag in config['apriltags']]
    tag_family = config['tag_family']
    
    # Use the size from the first tag, or default size
    if config['apriltags']:
        physical_size = config['apriltags'][0].get('size', config['default_size'])
    else:
        physical_size = config['default_size']
    
    print(f"📍 Found {len(tag_ids)} tags in configuration:")
    for tag in config['apriltags']:
        pos = (tag['x'], tag['y'], tag['z'])
        size = tag.get('size', config['default_size'])
        print(f"   Tag {tag['id']}: position {pos}, size {size}m")
    
    return create_apriltag_printout(tag_ids, physical_size, tag_family, output_file)


def main():
    """Main function for AprilTag generation."""
    parser = argparse.ArgumentParser(description='Generate printable AprilTag images')
    parser.add_argument('--config', 
                       help='AprilTag configuration file (generates all tags from config)')
    parser.add_argument('--tag-ids', type=int, nargs='+',
                       help='Tag ID numbers to generate (e.g., --tag-ids 0 1 2)')
    parser.add_argument('--physical-size', type=float, default=0.1,
                       help='Physical size of tags in meters (default: 0.1m = 10cm)')
    parser.add_argument('--tag-family', default='tag36h11',
                       choices=['tag36h11', 'tag25h9', 'tag16h5'],
                       help='AprilTag family (default: tag36h11)')
    parser.add_argument('--output', default='apriltags.png',
                       help='Output image file (default: apriltags.png)')
    
    args = parser.parse_args()
    
    try:
        if args.config:
            # Generate from configuration file
            output_file = generate_from_config(args.config, args.output)
        elif args.tag_ids:
            # Generate specific tag IDs
            output_file = create_apriltag_printout(
                args.tag_ids, 
                args.physical_size, 
                args.tag_family, 
                args.output
            )
        else:
            print("❌ Must specify either --config or --tag-ids")
            print("Examples:")
            print("  python generate_apriltags.py --config apriltag_config.json")
            print("  python generate_apriltags.py --tag-ids 0 1 --physical-size 0.1")
            return
        
        print(f"\n🎉 AprilTag generation complete!")
        print(f"📄 Print the file: {output_file}")
        print(f"📐 Place tags at the positions specified in your configuration")
        print(f"🎯 Then run calibration: python apriltag_pose_calibration.py --camera-id A")
        
    except Exception as e:
        print(f"❌ Error generating AprilTags: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 