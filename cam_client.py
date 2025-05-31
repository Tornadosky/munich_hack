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
    
    def __init__(self, pose_file: str, server_addr: str, target_class: str, fps: float):
        """
        Initialize camera client with configuration.
        
        Args:
            pose_file: Path to camera pose JSON file
            server_addr: Server address as "host:port"
            target_class: YOLO class name to detect (e.g., "bottle")
            fps: Detection frame rate
        """
        # Load camera pose configuration
        with open(pose_file, 'r') as f:
            self.pose = json.load(f)
        
        # Parse server address
        host, port = server_addr.split(':')
        self.server_addr = (host, int(port))
        
        self.target_class = target_class
        self.detection_interval = 1.0 / fps
        
        # Initialize YOLOv8-nano model
        print("Loading YOLOv8-nano model...")
        self.model = YOLO('yolov8n.pt')
        
        # Map COCO class names to indices
        self.class_names = self.model.names
        self.target_idx = None
        for idx, name in self.class_names.items():
            if name.lower() == target_class.lower():
                self.target_idx = idx
                break
        
        if self.target_idx is None:
            raise ValueError(f"Target class '{target_class}' not found in YOLO classes")
        
        print(f"Camera {self.pose['cam_id']} ready, detecting '{target_class}'")
        print(f"Will send detections to {self.server_addr}")
        
        # Test UDP connection
        self.test_connection()
    
    def test_connection(self):
        """Test UDP connection to the server."""
        test_packet = {
            "cam_id": self.pose['cam_id'],
            "test": True,
            "timestamp": time.time()
        }
        
        print(f"Testing UDP connection to {self.server_addr}...")
        success = udp_send_with_retry(test_packet, self.server_addr, max_retries=1)
        if success:
            print("✓ UDP connection test successful")
        else:
            print("✗ UDP connection test failed - check server address and network")
    
    def start_capture(self):
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Set capture resolution to match pose config
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.pose['img_w'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.pose['img_h'])
        
        # Verify actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h} (requested: {self.pose['img_w']}x{self.pose['img_h']})")
    
    def detect_objects(self, frame):
        """
        Run YOLO detection on frame and return target bounding boxes.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            List of (cx, cy) center coordinates for target detections
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
                        
                        # Calculate center point
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        detections.append((cx, cy))
        
        return detections
    
    def send_detection(self, cx: float, cy: float) -> bool:
        """
        Send detection packet to fusion server via UDP.
        
        Args:
            cx, cy: Center coordinates of detected object
            
        Returns:
            True if packet was sent successfully
        """
        packet = {
            "cam_id": self.pose['cam_id'],
            "cx": cx,
            "cy": cy,
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
            print(f"✓ Sent detection: cam={self.pose['cam_id']}, cx={cx:.1f}, cy={cy:.1f}")
        else:
            print(f"✗ Failed to send detection: cam={self.pose['cam_id']}, cx={cx:.1f}, cy={cy:.1f}")
        
        return success
    
    async def run_detection_loop(self):
        """Main detection loop running at specified FPS."""
        self.start_capture()
        
        frame_count = 0
        successful_sends = 0
        failed_sends = 0
        
        try:
            print(f"Starting detection loop at {1/self.detection_interval:.1f} FPS")
            print("Press 'q' to quit")
            
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Run object detection
                detections = self.detect_objects(frame)
                
                # Send each detection via UDP
                for cx, cy in detections:
                    success = self.send_detection(cx, cy)
                    if success:
                        successful_sends += 1
                    else:
                        failed_sends += 1
                
                # Display frame with detection count and stats
                status_text = f"Detections: {len(detections)} | Sent: {successful_sends} | Failed: {failed_sends}"
                cv2.putText(frame, status_text, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame rate
                fps_text = f"Frame: {frame_count}"
                cv2.putText(frame, fps_text,
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f"Camera {self.pose['cam_id']}", frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Maintain FPS timing
                elapsed = time.time() - start_time
                if elapsed < self.detection_interval:
                    await asyncio.sleep(self.detection_interval - elapsed)
                    
        finally:
            print(f"\nSession stats:")
            print(f"Frames processed: {frame_count}")
            print(f"Successful sends: {successful_sends}")
            print(f"Failed sends: {failed_sends}")
            
            self.cap.release()
            cv2.destroyAllWindows()


async def main():
    """Parse arguments and run camera client."""
    parser = argparse.ArgumentParser(description='Camera client for object detection')
    parser.add_argument('--pose', required=True, help='Camera pose JSON file')
    parser.add_argument('--server', required=True, help='Server address (host:port)')
    parser.add_argument('--target', default='bottle', help='Target object class')
    parser.add_argument('--fps', type=float, default=2.0, help='Detection FPS')
    
    args = parser.parse_args()
    
    # Create and run client
    try:
        client = CameraClient(args.pose, args.server, args.target, args.fps)
        await client.run_detection_loop()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 