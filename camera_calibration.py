"""
Camera Calibration Tool for Multi-Camera Triangulation System

This tool helps you:
1. Measure real-world positions and orientations of your laptops
2. Calibrate camera field of view (FOV)
3. Generate accurate pose files for triangulation
4. Test triangulation accuracy with known reference points

Usage:
    python camera_calibration.py --mode measure     # Interactive measurement guide
    python camera_calibration.py --mode fov        # FOV calibration
    python camera_calibration.py --mode generate   # Generate pose files
    python camera_calibration.py --mode test       # Test calibration accuracy
"""

import cv2
import json
import math
import numpy as np
import argparse
import time
from pathlib import Path


class CameraCalibrator:
    """Tool for measuring camera positions and calibrating triangulation setup."""
    
    def __init__(self):
        """Initialize calibrator with default camera settings."""
        # Standard camera parameters (will be updated during calibration)
        self.default_fov = 70.0  # degrees, typical webcam FOV
        self.img_width = 640
        self.img_height = 480
        
        # Measurement data storage
        self.cameras = {}
        self.reference_points = []
        
    def interactive_measurement_guide(self):
        """Interactive guide to help measure camera positions and orientations."""
        print("🎯 Camera Position & Orientation Measurement Guide")
        print("=" * 60)
        print()
        
        # Step 1: Choose coordinate system
        print("STEP 1: Set up your coordinate system")
        print("-" * 40)
        print("Choose a corner of your room as origin (0, 0)")
        print("• X-axis: horizontal (left-right)")
        print("• Y-axis: vertical (forward-back)")
        print("• Positive X = right, Positive Y = forward")
        print()
        
        origin_description = input("📍 Describe your chosen origin point: ")
        print(f"✅ Origin set: {origin_description}")
        print()
        
        # Step 2: Measure each camera
        num_cameras = int(input("How many cameras (laptops) do you have? "))
        
        for i in range(num_cameras):
            camera_id = input(f"Enter ID for camera {i+1} (e.g., A, B, laptop1): ").strip()
            print(f"\n📷 Measuring Camera {camera_id}")
            print("-" * 30)
            
            # Position measurement
            print("Position measurement:")
            print("• Use a measuring tape from your origin point")
            print("• Measure to the CENTER of the laptop screen")
            
            x = float(input(f"  X coordinate (meters, right from origin): "))
            y = float(input(f"  Y coordinate (meters, forward from origin): "))
            
            # Orientation measurement
            print("\nOrientation measurement:")
            print("• Yaw = direction the camera is pointing")
            print("• 0° = East (positive X direction)")
            print("• 90° = North (positive Y direction)")
            print("• 180° = West (negative X direction)")
            print("• 270° = South (negative Y direction)")
            
            # Visual aid for orientation
            print("\nVisual reference:")
            print("  North (90°)")
            print("      ↑")
            print("West ← + → East (0°)")
            print("      ↓")
            print("  South (270°)")
            
            yaw = float(input(f"  Camera yaw angle (degrees): "))
            
            # Store camera data
            self.cameras[camera_id] = {
                'cam_id': camera_id,
                'x': x,
                'y': y,
                'yaw_deg': yaw,
                'fov_deg': self.default_fov,  # Will calibrate separately
                'img_w': self.img_width,
                'img_h': self.img_height
            }
            
            print(f"✅ Camera {camera_id} measured: ({x:.1f}, {y:.1f}) facing {yaw:.0f}°")
        
        # Step 3: Add reference points
        print(f"\n📐 Reference Points for Accuracy Testing")
        print("-" * 40)
        print("Add some known reference points to test triangulation accuracy")
        
        add_refs = input("Do you want to add reference points? (y/n): ").lower() == 'y'
        
        if add_refs:
            while True:
                ref_name = input("Reference point name (or 'done'): ").strip()
                if ref_name.lower() == 'done':
                    break
                
                ref_x = float(input(f"  {ref_name} X coordinate: "))
                ref_y = float(input(f"  {ref_name} Y coordinate: "))
                
                self.reference_points.append({
                    'name': ref_name,
                    'x': ref_x,
                    'y': ref_y
                })
                
                print(f"✅ Added reference: {ref_name} at ({ref_x:.1f}, {ref_y:.1f})")
        
        # Save measurement data
        self.save_measurements()
        print(f"\n✅ Measurements saved to 'calibration_data.json'")
        print("💡 Next: Run FOV calibration with --mode fov")
    
    def calibrate_field_of_view(self):
        """Interactive FOV calibration using a known object width."""
        print("🔍 Field of View (FOV) Calibration")
        print("=" * 40)
        print()
        
        # Load existing measurements
        if not self.load_measurements():
            print("❌ No measurement data found. Run --mode measure first.")
            return
        
        print("FOV calibration using known object method:")
        print("1. Place a known-width object (e.g., bottle, book) at known distance")
        print("2. Measure how many pixels it covers horizontally")
        print("3. Calculate FOV from geometry")
        print()
        
        # Get calibration object info
        object_width = float(input("Object width in real world (meters): "))
        distance = float(input("Distance from camera to object (meters): "))
        
        # Start camera for measurement
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        
        print(f"\n📷 Camera started. Instructions:")
        print("1. Point camera at your calibration object")
        print("2. Press 'c' to capture measurement")
        print("3. Click left and right edges of the object")
        print("4. Press 'q' to finish")
        
        measuring = True
        pixel_width = None
        
        while measuring:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw crosshairs for alignment
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
            
            # Add instructions
            cv2.putText(frame, "Position object, press 'c' to measure", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if pixel_width:
                cv2.putText(frame, f"Measured: {pixel_width:.1f} pixels", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('FOV Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Capture for measurement
                pixel_width = self.measure_object_pixels(frame.copy())
                if pixel_width:
                    print(f"📏 Measured object width: {pixel_width:.1f} pixels")
            elif key == ord('q'):
                measuring = False
        
        cap.release()
        cv2.destroyAllWindows()
        
        if pixel_width:
            # Calculate FOV from measurements
            # Angular width of object = 2 * atan(object_width / (2 * distance))
            angular_width_rad = 2 * math.atan(object_width / (2 * distance))
            angular_width_deg = math.degrees(angular_width_rad)
            
            # FOV = angular_width * (image_width / pixel_width)
            calculated_fov = angular_width_deg * (self.img_width / pixel_width)
            
            print(f"\n📊 FOV Calculation Results:")
            print(f"  Object angular width: {angular_width_deg:.2f}°")
            print(f"  Pixel coverage: {pixel_width:.1f} px")
            print(f"  Calculated FOV: {calculated_fov:.1f}°")
            
            # Update all cameras with calibrated FOV
            for camera in self.cameras.values():
                camera['fov_deg'] = calculated_fov
            
            self.save_measurements()
            print(f"✅ FOV updated for all cameras: {calculated_fov:.1f}°")
        else:
            print("❌ No measurement taken")
    
    def measure_object_pixels(self, frame):
        """Interactive pixel measurement of object width."""
        print("Click left edge, then right edge of the object...")
        
        clicks = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicks.append(x)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Measurement', frame)
                
                if len(clicks) == 2:
                    # Draw line between points
                    cv2.line(frame, (clicks[0], y), (clicks[1], y), (0, 255, 0), 2)
                    pixel_width = abs(clicks[1] - clicks[0])
                    cv2.putText(frame, f"{pixel_width:.1f} pixels", 
                               (min(clicks), y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Measurement', frame)
        
        cv2.imshow('Measurement', frame)
        cv2.setMouseCallback('Measurement', mouse_callback)
        
        # Wait for two clicks
        while len(clicks) < 2:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow('Measurement')
        
        return abs(clicks[1] - clicks[0]) if len(clicks) == 2 else None
    
    def generate_pose_files(self):
        """Generate pose JSON files from calibration data."""
        print("📄 Generating Pose Files")
        print("=" * 30)
        
        if not self.load_measurements():
            print("❌ No measurement data found. Run --mode measure first.")
            return
        
        # Create pose files for each camera
        for cam_id, camera_data in self.cameras.items():
            filename = f"pose_{cam_id}.json"
            
            with open(filename, 'w') as f:
                json.dump(camera_data, f, indent=2)
            
            print(f"✅ Created {filename}")
            print(f"   Position: ({camera_data['x']:.1f}, {camera_data['y']:.1f})")
            print(f"   Orientation: {camera_data['yaw_deg']:.1f}°")
            print(f"   FOV: {camera_data['fov_deg']:.1f}°")
            print()
        
        print("💡 Pose files ready! You can now run your triangulation system.")
    
    def test_calibration_accuracy(self):
        """Test triangulation accuracy using reference points."""
        print("🧪 Calibration Accuracy Test")
        print("=" * 35)
        
        if not self.load_measurements():
            print("❌ No measurement data found. Run --mode measure first.")
            return
        
        if len(self.cameras) < 2:
            print("❌ Need at least 2 cameras for triangulation test.")
            return
        
        if not self.reference_points:
            print("❌ No reference points defined. Add some with --mode measure.")
            return
        
        from utils import ray_from_pixel, triangulate_two_rays
        
        print("Manual triangulation test:")
        print("1. Place a bottle at one of your reference points")
        print("2. Note the pixel coordinates in each camera view")
        print("3. Compare triangulated position with known position")
        print()
        
        # Show reference points
        print("Available reference points:")
        for i, ref in enumerate(self.reference_points):
            print(f"  {i+1}. {ref['name']}: ({ref['x']:.1f}, {ref['y']:.1f})")
        
        ref_idx = int(input("Choose reference point number: ")) - 1
        ref_point = self.reference_points[ref_idx]
        
        print(f"\n📍 Testing with: {ref_point['name']} at ({ref_point['x']:.1f}, {ref_point['y']:.1f})")
        
        # Get camera list
        cam_ids = list(self.cameras.keys())
        print(f"\nAvailable cameras: {', '.join(cam_ids)}")
        
        cam1_id = input("First camera ID: ").strip()
        cam2_id = input("Second camera ID: ").strip()
        
        if cam1_id not in self.cameras or cam2_id not in self.cameras:
            print("❌ Invalid camera IDs")
            return
        
        # Get pixel coordinates
        print(f"\nPlace bottle at {ref_point['name']} and observe pixel coordinates:")
        
        cam1_cx = float(input(f"  Camera {cam1_id} - bottle center X pixel: "))
        cam2_cx = float(input(f"  Camera {cam2_id} - bottle center X pixel: "))
        
        # Perform triangulation
        try:
            # Convert pixels to rays
            pose1 = self.cameras[cam1_id]
            pose2 = self.cameras[cam2_id]
            
            ray1 = ray_from_pixel(cam1_cx, pose1)
            ray2 = ray_from_pixel(cam2_cx, pose2)
            
            pos1 = np.array([pose1['x'], pose1['y']])
            pos2 = np.array([pose2['x'], pose2['y']])
            
            # Triangulate
            calculated_pos = triangulate_two_rays(pos1, ray1, pos2, ray2)
            
            # Calculate error
            error_x = calculated_pos[0] - ref_point['x']
            error_y = calculated_pos[1] - ref_point['y']
            total_error = math.sqrt(error_x**2 + error_y**2)
            
            print(f"\n📊 Triangulation Results:")
            print(f"  Expected: ({ref_point['x']:.2f}, {ref_point['y']:.2f})")
            print(f"  Calculated: ({calculated_pos[0]:.2f}, {calculated_pos[1]:.2f})")
            print(f"  Error: ({error_x:.2f}, {error_y:.2f})")
            print(f"  Total error: {total_error:.2f} meters")
            
            if total_error < 0.3:
                print("✅ Good accuracy (< 30cm error)")
            elif total_error < 0.5:
                print("⚠️  Moderate accuracy (< 50cm error)")
            else:
                print("❌ Poor accuracy (> 50cm error)")
                print("💡 Consider recalibrating FOV or double-checking measurements")
            
        except Exception as e:
            print(f"❌ Triangulation failed: {e}")
    
    def save_measurements(self):
        """Save calibration data to JSON file."""
        data = {
            'cameras': self.cameras,
            'reference_points': self.reference_points,
            'image_resolution': {
                'width': self.img_width,
                'height': self.img_height
            }
        }
        
        with open('calibration_data.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_measurements(self):
        """Load calibration data from JSON file."""
        try:
            with open('calibration_data.json', 'r') as f:
                data = json.load(f)
            
            self.cameras = data.get('cameras', {})
            self.reference_points = data.get('reference_points', [])
            
            resolution = data.get('image_resolution', {})
            self.img_width = resolution.get('width', 640)
            self.img_height = resolution.get('height', 480)
            
            return True
        except FileNotFoundError:
            return False


def main():
    """Main function with mode selection."""
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--mode', choices=['measure', 'fov', 'generate', 'test'],
                       required=True, help='Calibration mode')
    
    args = parser.parse_args()
    calibrator = CameraCalibrator()
    
    if args.mode == 'measure':
        calibrator.interactive_measurement_guide()
    elif args.mode == 'fov':
        calibrator.calibrate_field_of_view()
    elif args.mode == 'generate':
        calibrator.generate_pose_files()
    elif args.mode == 'test':
        calibrator.test_calibration_accuracy()


if __name__ == "__main__":
    main() 