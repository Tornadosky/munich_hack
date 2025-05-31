"""
Camera client for object detection and UDP transmission.
Runs YOLOv8 detection on webcam feed and sends target detections to fusion server.
"""

import cv2
import json
import time
import argparse
import asyncio
from ultralytics import YOLO
from utils import udp_send


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
    
    def start_capture(self):
        """Initialize webcam capture."""
        # Try to find a working camera
        for camera_id in range(5):
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    print(f"Using camera {camera_id}")
                    break
                self.cap.release()
        else:
            raise RuntimeError("Cannot find any working webcam")
        
        # Set capture resolution to match pose config
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.pose['img_w'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.pose['img_h'])
    
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
    
    def send_detection(self, cx: float, cy: float):
        """
        Send detection packet to fusion server via UDP.
        
        Args:
            cx, cy: Center coordinates of detected object
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
        
        udp_send(packet, self.server_addr)
        print(f"Sent detection: cam={self.pose['cam_id']}, cx={cx:.1f}, cy={cy:.1f}")
    
    async def run_detection_loop(self):
        """Main detection loop running at specified FPS."""
        self.start_capture()
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Run object detection
                detections = self.detect_objects(frame)
                
                # Send each detection via UDP
                for cx, cy in detections:
                    self.send_detection(cx, cy)
                
                # Display frame with detection count
                cv2.putText(frame, f"Detections: {len(detections)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f"Camera {self.pose['cam_id']}", frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Maintain FPS timing
                elapsed = time.time() - start_time
                if elapsed < self.detection_interval:
                    await asyncio.sleep(self.detection_interval - elapsed)
                    
        finally:
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
    client = CameraClient(args.pose, args.server, args.target, args.fps)
    await client.run_detection_loop()


if __name__ == "__main__":
    asyncio.run(main()) 