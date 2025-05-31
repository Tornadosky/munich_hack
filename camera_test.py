"""
Simple camera test script to diagnose webcam access issues.
Tests different camera indices and provides troubleshooting info.
"""

import cv2
import sys


def test_camera_access():
    """Test camera access with different methods and indices."""
    print("🎥 Camera Access Diagnostic Tool")
    print("=" * 50)
    
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
                print(f"   📏 Resolution: {width}x{height}")
                print(f"   📱 Frame shape: {frame.shape}")
                
                # Show frame for 2 seconds
                cv2.imshow(f'Camera {camera_id} Test', frame)
                print("   💡 Showing test frame for 2 seconds...")
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                
                cap.release()
                print(f"   ✅ Camera {camera_id}: Working perfectly!")
                return camera_id
            else:
                print(f"   ❌ Camera {camera_id}: Cannot read frames")
                cap.release()
        else:
            print(f"   ❌ Camera {camera_id}: Cannot open")
    
    print("\n" + "=" * 50)
    print("❌ No working cameras found!")
    return None


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
    }
    
    print("Camera Properties:")
    for name, prop in properties.items():
        try:
            value = cap.get(prop)
            print(f"   {name}: {value}")
        except Exception as e:
            print(f"   {name}: Error - {e}")
    
    cap.release()


def run_live_test(camera_id):
    """Run a live camera feed test."""
    print(f"\n📹 Starting live test for camera {camera_id}")
    print("Press 'q' to quit, 's' to save screenshot")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("❌ Cannot open camera for live test")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"❌ Failed to read frame {frame_count}")
                break
            
            frame_count += 1
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Live Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"camera_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Saved: {filename}")
                
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Live test completed. Total frames: {frame_count}")


def main():
    """Main diagnostic function."""
    print("Starting camera diagnostics...\n")
    
    # Test basic camera access
    working_camera = test_camera_access()
    
    if working_camera is None:
        print("\n🚨 Troubleshooting Tips:")
        print("1. Check if camera is physically connected")
        print("2. Close other apps that might be using the camera (Zoom, Teams, etc.)")
        print("3. Check Windows Privacy Settings:")
        print("   Settings > Privacy > Camera > Allow apps to access camera")
        print("4. Try running as administrator")
        print("5. Update camera drivers")
        print("6. Try different USB ports")
        sys.exit(1)
    
    # Test properties
    test_camera_properties(working_camera)
    
    # Ask user if they want live test
    response = input(f"\n🤔 Run live test for camera {working_camera}? (y/n): ").lower()
    if response == 'y':
        run_live_test(working_camera)
    
    print(f"\n🎉 Camera {working_camera} is ready for YOLO detection!")
    print(f"💡 Use this camera ID in your scripts: {working_camera}")


if __name__ == "__main__":
    main() 