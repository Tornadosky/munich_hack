"""
Enhanced camera test script to diagnose webcam access issues and detect optimal settings.
Tests different camera indices, resolutions, and provides troubleshooting info.
"""

import cv2
import sys
import json


def test_camera_access():
    """Test camera access with different methods and indices."""
    print("🎥 Camera Access Diagnostic Tool")
    print("=" * 50)
    
    working_cameras = []
    
    # Test multiple camera indices
    for camera_id in range(5):
        print(f"\nTesting camera index {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            print(f"✅ Camera {camera_id}: Successfully opened")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"   📏 Default resolution: {width}x{height}")
                print(f"   📱 Frame shape: {frame.shape}")
                
                # Test different resolutions
                test_resolutions = [
                    (640, 480),   # VGA
                    (1280, 720),  # HD
                    (1920, 1080), # Full HD
                    (320, 240),   # QVGA
                ]
                
                print(f"   🔧 Testing resolution support:")
                supported_resolutions = []
                
                for test_w, test_h in test_resolutions:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_h)
                    
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if (actual_w, actual_h) == (test_w, test_h):
                        print(f"      ✅ {test_w}x{test_h}: Supported")
                        supported_resolutions.append((test_w, test_h))
                    else:
                        print(f"      ❌ {test_w}x{test_h}: Not supported (got {actual_w}x{actual_h})")
                
                # Reset to default resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                working_cameras.append({
                    'id': camera_id,
                    'default_resolution': (width, height),
                    'supported_resolutions': supported_resolutions
                })
                
                # Show frame for 2 seconds
                cv2.imshow(f'Camera {camera_id} Test', frame)
                print("   💡 Showing test frame for 2 seconds...")
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                cap.release()
                print(f"   ✅ Camera {camera_id}: Working perfectly!")
            else:
                print(f"   ❌ Camera {camera_id}: Cannot read frames")
                cap.release()
        else:
            print(f"   ❌ Camera {camera_id}: Cannot open")
    
    print("\n" + "=" * 50)
    if working_cameras:
        print(f"✅ Found {len(working_cameras)} working camera(s)!")
        return working_cameras
    else:
        print("❌ No working cameras found!")
        return []


def test_camera_properties(camera_id):
    """Test and display camera properties."""
    print(f"\n🔍 Testing camera {camera_id} properties...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Cannot open camera for property testing")
        return
    
    # Test common properties
    properties = {
        'Frame Width': cv2.CAP_PROP_FRAME_WIDTH,
        'Frame Height': cv2.CAP_PROP_FRAME_HEIGHT,
        'FPS': cv2.CAP_PROP_FPS,
        'Format': cv2.CAP_PROP_FORMAT,
        'Backend': cv2.CAP_PROP_BACKEND,
        'Brightness': cv2.CAP_PROP_BRIGHTNESS,
        'Contrast': cv2.CAP_PROP_CONTRAST,
        'Saturation': cv2.CAP_PROP_SATURATION,
        'Auto Exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
        'Exposure': cv2.CAP_PROP_EXPOSURE,
    }
    
    print("Camera Properties:")
    for name, prop in properties.items():
        try:
            value = cap.get(prop)
            print(f"   {name}: {value}")
        except Exception as e:
            print(f"   {name}: Error - {e}")
    
    cap.release()


def generate_pose_files(working_cameras):
    """Generate pose files based on detected camera capabilities."""
    if not working_cameras:
        print("❌ No working cameras to generate poses for")
        return
    
    print(f"\n📄 Generating pose files for {len(working_cameras)} camera(s)...")
    
    # Default camera positions (user should adjust these)
    default_positions = [
        {'x': 0.0, 'y': 0.0, 'yaw_deg': 45.0},   # Camera A
        {'x': 4.0, 'y': 0.0, 'yaw_deg': 135.0},  # Camera B
        {'x': 2.0, 'y': 4.0, 'yaw_deg': 225.0},  # Camera C
        {'x': 0.0, 'y': 4.0, 'yaw_deg': 315.0},  # Camera D
    ]
    
    camera_ids = ['A', 'B', 'C', 'D']
    
    for i, camera_info in enumerate(working_cameras):
        if i >= len(camera_ids):
            break
            
        cam_id = camera_ids[i]
        default_res = camera_info['default_resolution']
        
        # Use 640x480 if supported, otherwise use default resolution
        if (640, 480) in camera_info['supported_resolutions']:
            img_w, img_h = 640, 480
            print(f"   Using 640x480 for camera {cam_id}")
        else:
            img_w, img_h = default_res
            print(f"   Using default resolution {img_w}x{img_h} for camera {cam_id}")
        
        pose = {
            'cam_id': cam_id,
            'x': default_positions[i]['x'],
            'y': default_positions[i]['y'],
            'yaw_deg': default_positions[i]['yaw_deg'],
            'fov_deg': 70.0,  # Typical webcam FOV
            'img_w': img_w,
            'img_h': img_h,
            'camera_index': camera_info['id']  # Store the actual camera index
        }
        
        filename = f"pose_{cam_id}.json"
        with open(filename, 'w') as f:
            json.dump(pose, f, indent=2)
        
        print(f"   ✅ Created {filename}")
        print(f"      Camera index: {camera_info['id']}")
        print(f"      Resolution: {img_w}x{img_h}")
        print(f"      Position: ({pose['x']}, {pose['y']}) facing {pose['yaw_deg']}°")
    
    print(f"\n💡 Pose files created! Remember to:")
    print(f"   1. Measure actual camera positions and update the x, y, yaw_deg values")
    print(f"   2. Calibrate the FOV using camera_calibration.py")
    print(f"   3. Test with: python cam_client.py --pose pose_A.json --server localhost:9000")


def interactive_camera_test():
    """Interactive camera testing with live preview."""
    working_cameras = test_camera_access()
    
    if not working_cameras:
        return
    
    print(f"\n🎮 Interactive Camera Test")
    print("=" * 30)
    
    # Let user choose camera
    if len(working_cameras) == 1:
        chosen_camera = working_cameras[0]
        print(f"Using the only available camera: {chosen_camera['id']}")
    else:
        print("Available cameras:")
        for i, cam in enumerate(working_cameras):
            print(f"   {i}: Camera {cam['id']} - {cam['default_resolution'][0]}x{cam['default_resolution'][1]}")
        
        choice = int(input("Choose camera number: "))
        chosen_camera = working_cameras[choice]
    
    camera_id = chosen_camera['id']
    print(f"\n📹 Starting interactive test with camera {camera_id}")
    print("Press 'q' to quit, 'p' to test properties")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame")
                break
            
            frame_count += 1
            
            # Add info overlay
            height, width = frame.shape[:2]
            cv2.putText(frame, f"Camera {camera_id} - Frame {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 'p' for properties", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw crosshairs
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 255), 1)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 255), 1)
            
            cv2.imshow(f'Camera {camera_id} Interactive Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print(f"\n📊 Current camera properties:")
                test_camera_properties(camera_id)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"📈 Captured {frame_count} frames total")


def main():
    """Main function with menu options."""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_camera_test()
        elif sys.argv[1] == '--generate-poses':
            working_cameras = test_camera_access()
            generate_pose_files(working_cameras)
        else:
            print("Usage: python camera_test.py [--interactive|--generate-poses]")
    else:
        # Default: run basic test and generate poses
        working_cameras = test_camera_access()
        if working_cameras:
            generate_pose_files(working_cameras)
            print(f"\n🎯 Next steps:")
            print(f"   1. Run: python camera_test.py --interactive")
            print(f"   2. Test detection: python cam_client.py --pose pose_A.json --server localhost:9000")


if __name__ == "__main__":
    main() 