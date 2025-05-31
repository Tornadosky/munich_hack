"""
Camera client for object detection and UDP transmission.
Runs YOLOv8 detection on webcam feed and sends target detections to fusion server.
"""

import cv2
import json
import time
import argparse
import asyncio
import socket
from ultralytics import YOLO


def udp_send_with_retry(payload: dict, addr: tuple, max_retries: int = 3) -> bool:
    """
    Send JSON payload via UDP with retry logic and better error handling.
    
    Args:
        payload: Dictionary to send as JSON
        addr: (host, port) tuple for destination
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(1.0)  # 1 second timeout
            message = json.dumps(payload).encode('utf-8')
            sock.sendto(message, addr)
            sock.close()
            return True
        except socket.timeout:
            print(f"UDP send timeout (attempt {attempt + 1}/{max_retries})")
        except socket.gaierror as e:
            print(f"UDP send DNS error: {e}")
            return False  # Don't retry DNS errors
        except Exception as e:
            print(f"UDP send error (attempt {attempt + 1}/{max_retries}): {e}")
        finally:
            try:
                sock.close()
            except:
                pass
    
    print(f"Failed to send UDP packet after {max_retries} attempts")
    return False


class CameraClient:
    """Webcam object detection client with UDP transmission."""
    
    def __init__(self, pose_file: str, server_addr: str, target_class: str, fps: float, flip_horizontal: bool = True):
        """
        Initialize camera client with configuration.
        
        Args:
            pose_file: Path to camera pose JSON file
            server_addr: Server address as "host:port"
            target_class: YOLO class name to detect (e.g., "bottle")
            fps: Detection frame rate
            flip_horizontal: Whether to flip the camera feed horizontally (default True)
        """
        # Load camera pose configuration
        with open(pose_file, 'r') as f:
            self.pose = json.load(f)
        
        print(f"📋 Loaded pose for camera {self.pose['cam_id']}:")
        print(f"   Position: ({self.pose['x']}, {self.pose['y']})")
        print(f"   Orientation: {self.pose['yaw_deg']}°")
        print(f"   FOV: {self.pose['fov_deg']}°")
        print(f"   Expected resolution: {self.pose['img_w']}x{self.pose['img_h']}")
        
        # Parse server address
        host, port = server_addr.split(':')
        self.server_addr = (host, int(port))
        
        self.target_class = target_class
        self.detection_interval = 1.0 / fps
        self.flip_horizontal = flip_horizontal
        
        print(f"🔄 Horizontal flip: {'Enabled' if flip_horizontal else 'Disabled'}")
        
        # Initialize YOLOv8-nano model
        print("📦 Loading YOLOv8-nano model...")
        self.model = YOLO('yolov8n.pt')
        
        # Map COCO class names to indices
        self.class_names = self.model.names
        self.target_idx = None
        for idx, name in self.class_names.items():
            if name.lower() == target_class.lower():
                self.target_idx = idx
                break
        
        if self.target_idx is None:
            available_classes = [name for name in self.class_names.values()]
            print(f"❌ Target class '{target_class}' not found in YOLO classes")
            print(f"Available classes: {', '.join(sorted(available_classes))}")
            raise ValueError(f"Target class '{target_class}' not found in YOLO classes")
        
        print(f"✅ Camera {self.pose['cam_id']} ready, detecting '{target_class}' (class index {self.target_idx})")
        print(f"🌐 Will send detections to {self.server_addr}")
        
        # Test UDP connection
        self.test_connection()
        
        # Camera capture object (will be initialized later)
        self.cap = None
        self.actual_resolution = None
    
    def test_connection(self):
        """Test UDP connection to the server."""
        test_packet = {
            "cam_id": self.pose['cam_id'],
            "test": True,
            "timestamp": time.time(),
            "pose": self.pose  # Include pose info in test
        }
        
        print(f"🔗 Testing UDP connection to {self.server_addr}...")
        success = udp_send_with_retry(test_packet, self.server_addr, max_retries=1)
        if success:
            print("✅ UDP connection test successful")
        else:
            print("❌ UDP connection test failed - check server address and network")
    
    def find_working_camera(self):
        """Find and test camera indices to get the best working camera."""
        print("🔍 Searching for working camera...")
        
        for camera_id in range(5):  # Test camera indices 0-4
            print(f"   Testing camera index {camera_id}...")
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    print(f"   ✅ Camera {camera_id}: Working! Resolution: {width}x{height}")
                    cap.release()
                    return camera_id
                else:
                    print(f"   ❌ Camera {camera_id}: Opens but can't read frames")
            else:
                print(f"   ❌ Camera {camera_id}: Cannot open")
            
            cap.release()
        
        print("❌ No working cameras found!")
        return None
    
    def start_capture(self):
        """Initialize webcam capture with automatic camera detection."""
        # Find working camera
        camera_id = self.find_working_camera()
        if camera_id is None:
            raise RuntimeError("Cannot find any working webcam")
        
        print(f"📹 Using camera index {camera_id}")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Get current resolution before setting
        current_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📏 Current camera resolution: {current_w}x{current_h}")
        
        # Try to set capture resolution to match pose config
        print(f"🎯 Attempting to set resolution to {self.pose['img_w']}x{self.pose['img_h']}...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.pose['img_w'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.pose['img_h'])
        
        # Verify actual resolution after setting
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_resolution = (actual_w, actual_h)
        
        print(f"📐 Actual camera resolution: {actual_w}x{actual_h}")
        
        # Check if resolution matches expectations
        if (actual_w, actual_h) != (self.pose['img_w'], self.pose['img_h']):
            print(f"⚠️  Resolution mismatch!")
            print(f"   Expected: {self.pose['img_w']}x{self.pose['img_h']}")
            print(f"   Actual: {actual_w}x{actual_h}")
            print(f"   This may affect triangulation accuracy!")
            
            # Update pose with actual resolution
            self.pose['img_w'] = actual_w
            self.pose['img_h'] = actual_h
            print(f"   📝 Updated pose to use actual resolution")
        else:
            print(f"✅ Resolution matches pose configuration")
        
        # Test frame capture
        ret, test_frame = self.cap.read()
        if ret:
            print(f"✅ Test frame captured successfully: {test_frame.shape}")
        else:
            print(f"❌ Failed to capture test frame")
            raise RuntimeError("Cannot capture frames from camera")
    
    def detect_objects(self, frame):
        """
        Run YOLO detection on frame and return target bounding boxes.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            List of (cx, cy, confidence) tuples for target detections
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Check if detection matches target class
                    cls_idx = int(box.cls[0])
                    if cls_idx == self.target_idx:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        # Calculate center point
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        detections.append((cx, cy, confidence))
        
        return detections
    
    def send_detection(self, cx: float, cy: float, confidence: float) -> bool:
        """
        Send detection packet to fusion server via UDP.
        
        Args:
            cx, cy: Center coordinates of detected object
            confidence: Detection confidence score
            
        Returns:
            True if packet was sent successfully
        """
        packet = {
            "cam_id": self.pose['cam_id'],
            "cx": cx,
            "cy": cy,
            "confidence": confidence,
            "timestamp": time.time(),
            "pose": {
                "x": self.pose['x'],
                "y": self.pose['y'],
                "yaw_deg": self.pose['yaw_deg'],
                "fov_deg": self.pose['fov_deg'],
                "img_w": self.pose['img_w'],
                "img_h": self.pose['img_h']
            }
        }
        
        success = udp_send_with_retry(packet, self.server_addr)
        if success:
            print(f"✅ Sent detection: cam={self.pose['cam_id']}, cx={cx:.1f}, cy={cy:.1f}, conf={confidence:.2f}")
        else:
            print(f"❌ Failed to send detection: cam={self.pose['cam_id']}, cx={cx:.1f}, cy={cy:.1f}")
        
        return success
    
    async def run_detection_loop(self):
        """Main detection loop running at specified FPS."""
        self.start_capture()
        
        frame_count = 0
        detection_count = 0
        successful_sends = 0
        failed_sends = 0
        
        try:
            print(f"🚀 Starting detection loop at {1/self.detection_interval:.1f} FPS")
            print(f"🎯 Looking for '{self.target_class}' objects")
            print("📺 Press 'q' in the camera window to quit")
            print("=" * 60)
            
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                frame_count += 1
                
                # Apply horizontal flip if enabled
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)  # 1 = horizontal flip
                
                # Run object detection
                detections = self.detect_objects(frame)
                detection_count += len(detections)
                
                # Send each detection via UDP
                for cx, cy, confidence in detections:
                    success = self.send_detection(cx, cy, confidence)
                    if success:
                        successful_sends += 1
                    else:
                        failed_sends += 1
                
                # Create visualization frame
                display_frame = frame.copy()
                
                # Draw detections on frame
                for cx, cy, confidence in detections:
                    # Draw bounding box center
                    cv2.circle(display_frame, (int(cx), int(cy)), 10, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{confidence:.2f}", 
                               (int(cx) + 15, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display status information
                status_text = f"Detections: {len(detections)} | Total: {detection_count} | Sent: {successful_sends} | Failed: {failed_sends}"
                cv2.putText(display_frame, status_text, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show frame rate and camera info
                fps_text = f"Frame: {frame_count} | Cam: {self.pose['cam_id']} | Target: {self.target_class}"
                cv2.putText(display_frame, fps_text,
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show resolution info and flip status
                res_text = f"Resolution: {self.actual_resolution[0]}x{self.actual_resolution[1]} | Flip: {'ON' if self.flip_horizontal else 'OFF'}"
                cv2.putText(display_frame, res_text,
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw crosshairs for reference
                h, w = display_frame.shape[:2]
                cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 255), 1)
                cv2.line(display_frame, (0, h//2), (w, h//2), (0, 255, 255), 1)
                
                cv2.imshow(f"Camera {self.pose['cam_id']} - {self.target_class} Detection", display_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Print periodic status
                if frame_count % 30 == 0:  # Every 30 frames
                    print(f"📊 Status: {frame_count} frames, {detection_count} detections, {successful_sends} sent, {failed_sends} failed")
                
                # Maintain FPS timing
                elapsed = time.time() - start_time
                if elapsed < self.detection_interval:
                    await asyncio.sleep(self.detection_interval - elapsed)
                    
        finally:
            print(f"\n📈 Final Session Stats:")
            print(f"   Frames processed: {frame_count}")
            print(f"   Total detections: {detection_count}")
            print(f"   Successful sends: {successful_sends}")
            print(f"   Failed sends: {failed_sends}")
            if frame_count > 0:
                print(f"   Detection rate: {detection_count/frame_count:.2f} detections/frame")
            
            self.cap.release()
            cv2.destroyAllWindows()


async def main():
    """Parse arguments and run camera client."""
    parser = argparse.ArgumentParser(description='Camera client for object detection')
    parser.add_argument('--pose', required=True, help='Camera pose JSON file')
    parser.add_argument('--server', required=True, help='Server address (host:port)')
    parser.add_argument('--target', default='bottle', help='Target object class')
    parser.add_argument('--fps', type=float, default=2.0, help='Detection FPS')
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flip (enabled by default)')
    
    args = parser.parse_args()
    
    # Flip is enabled by default, disabled only if --no-flip is specified
    flip_horizontal = not args.no_flip
    
    # Create and run client
    try:
        client = CameraClient(args.pose, args.server, args.target, args.fps, flip_horizontal)
        await client.run_detection_loop()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 