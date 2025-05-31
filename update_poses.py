"""
Update pose files with actual camera resolutions.
This script detects the actual resolution of connected cameras and updates pose files accordingly.
"""

import cv2
import json
import os
from glob import glob


def detect_camera_resolution(camera_index=0):
    """
    Detect the actual resolution of a camera.
    
    Args:
        camera_index: Camera index to test
        
    Returns:
        (width, height) tuple or None if camera not accessible
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        
        # Try to read a frame to get actual resolution
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        
        height, width = frame.shape[:2]
        cap.release()
        return (width, height)
    
    except Exception as e:
        print(f"Error detecting camera {camera_index}: {e}")
        return None


def update_pose_files():
    """Update all pose files with detected camera resolutions."""
    print("🔍 Updating pose files with actual camera resolutions...")
    
    # Find all pose files
    pose_files = glob("pose_*.json")
    
    if not pose_files:
        print("❌ No pose files found (pose_*.json)")
        return
    
    print(f"📄 Found {len(pose_files)} pose files: {pose_files}")
    
    # Detect available cameras
    print("\n🎥 Detecting available cameras...")
    available_cameras = {}
    
    for camera_index in range(5):  # Test camera indices 0-4
        resolution = detect_camera_resolution(camera_index)
        if resolution:
            available_cameras[camera_index] = resolution
            print(f"   Camera {camera_index}: {resolution[0]}x{resolution[1]}")
    
    if not available_cameras:
        print("❌ No working cameras detected")
        return
    
    print(f"✅ Found {len(available_cameras)} working camera(s)")
    
    # Update pose files
    print(f"\n📝 Updating pose files...")
    
    for pose_file in pose_files:
        try:
            # Load existing pose
            with open(pose_file, 'r') as f:
                pose = json.load(f)
            
            print(f"\n   📄 Processing {pose_file}:")
            print(f"      Camera ID: {pose.get('cam_id', 'unknown')}")
            print(f"      Current resolution: {pose.get('img_w', '?')}x{pose.get('img_h', '?')}")
            
            # Check if pose has a specific camera index
            if 'camera_index' in pose and pose['camera_index'] in available_cameras:
                # Use the specified camera index
                camera_index = pose['camera_index']
                new_resolution = available_cameras[camera_index]
                print(f"      Using specified camera index {camera_index}: {new_resolution[0]}x{new_resolution[1]}")
            else:
                # Use the first available camera
                camera_index = list(available_cameras.keys())[0]
                new_resolution = available_cameras[camera_index]
                print(f"      Using first available camera {camera_index}: {new_resolution[0]}x{new_resolution[1]}")
                pose['camera_index'] = camera_index  # Store for future reference
            
            # Update resolution
            old_w, old_h = pose.get('img_w', 0), pose.get('img_h', 0)
            pose['img_w'], pose['img_h'] = new_resolution
            
            # Save updated pose
            with open(pose_file, 'w') as f:
                json.dump(pose, f, indent=2)
            
            if (old_w, old_h) != new_resolution:
                print(f"      ✅ Updated: {old_w}x{old_h} → {new_resolution[0]}x{new_resolution[1]}")
            else:
                print(f"      ✅ Confirmed: {new_resolution[0]}x{new_resolution[1]} (no change)")
        
        except Exception as e:
            print(f"      ❌ Error updating {pose_file}: {e}")
    
    print(f"\n🎯 Pose files updated! You can now run:")
    print(f"   python cam_client.py --pose pose_A.json --server localhost:9000")


if __name__ == "__main__":
    update_pose_files() 