"""
High-performance YOLO detection with smooth 60 FPS display.
Uses threading to separate camera capture from detection for lag-free experience.
"""

import cv2
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO


class SmoothYOLODetector:
    """High-performance YOLO detector with threading for smooth 60 FPS."""
    
    def __init__(self, model_name='yolov8n.pt', target_class=None, confidence=0.5):
        """
        Initialize smooth YOLO detector.
        
        Args:
            model_name: YOLO model to use
            target_class: Specific class to focus on (None for all)
            confidence: Detection confidence threshold
        """
        print("🚀 Initializing Smooth YOLO Detector")
        print("=" * 50)
        
        # Load YOLO model
        print("📦 Loading YOLO model...")
        self.model = YOLO(model_name)
        self.target_class = target_class.lower() if target_class else None
        self.confidence = confidence
        
        # Find target class index
        self.target_idx = None
        if self.target_class:
            for idx, name in self.model.names.items():
                if name.lower() == self.target_class:
                    self.target_idx = idx
                    break
        
        print(f"✅ Model loaded with {len(self.model.names)} classes")
        if self.target_class:
            print(f"🎯 Target class: {self.target_class}")
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid lag
        self.detection_queue = queue.Queue(maxsize=5)
        self.latest_detections = []
        self.running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.detection_fps = 0
        self.detection_counter = 0
        self.detection_start_time = time.time()
        
        # Find working camera
        self.camera_id = self._find_working_camera()
        if self.camera_id is None:
            raise RuntimeError("No working camera found!")
    
    def _find_working_camera(self):
        """Find first working camera index."""
        print("🔍 Finding working camera...")
        for camera_id in range(5):
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"✅ Using camera {camera_id}")
                    return camera_id
        return None
    
    def _camera_thread(self):
        """Camera capture thread - runs at maximum FPS."""
        cap = cv2.VideoCapture(self.camera_id)
        
        # Set camera properties for best performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        
        print(f"📹 Camera thread started (target 60 FPS)")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                # Skip frame if queue is full (prevents lag)
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put_nowait(frame.copy())  # Add new frame
                except queue.Empty:
                    pass
        
        cap.release()
        print("📹 Camera thread stopped")
    
    def _detection_thread(self):
        """Detection thread - runs at lower FPS for performance."""
        print("🔍 Detection thread started")
        
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame = self.frame_queue.get(timeout=0.1)
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                detections = self._process_results(results)
                
                # Update latest detections
                self.latest_detections = detections
                
                # Update detection FPS
                self.detection_counter += 1
                if self.detection_counter % 10 == 0:
                    current_time = time.time()
                    self.detection_fps = 10 / (current_time - self.detection_start_time)
                    self.detection_start_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
        
        print("🔍 Detection thread stopped")
    
    def _process_results(self, results):
        """Process YOLO results into simple detection list."""
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Filter by confidence and target class
                    if confidence < self.confidence:
                        continue
                    
                    if self.target_idx is not None and class_id != self.target_idx:
                        continue
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    })
        
        return detections
    
    def _draw_detections(self, frame):
        """Draw latest detections on frame."""
        annotated_frame = frame.copy()
        
        for detection in self.latest_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Choose color based on target class
            if self.target_idx is None or detection['class_id'] == self.target_idx:
                color = (0, 255, 0)  # Green for target
                thickness = 3
            else:
                color = (0, 0, 255)  # Red for others
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(annotated_frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def _draw_info_overlay(self, frame):
        """Draw performance info overlay."""
        # Update display FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            display_fps = 30 / (current_time - self.fps_start_time)
            self.display_fps = display_fps
            self.fps_start_time = current_time
        
        # Info text
        info_lines = [
            f"Display FPS: {getattr(self, 'display_fps', 0):.1f}",
            f"Detection FPS: {self.detection_fps:.1f}",
            f"Camera: {self.camera_id}",
            f"Detections: {len(self.latest_detections)}"
        ]
        
        if self.target_class:
            info_lines.append(f"Target: {self.target_class}")
        
        # Draw info with background
        for i, text in enumerate(info_lines):
            y_pos = 30 + i * 30
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background rectangle
            cv2.rectangle(frame, (5, y_pos - 25), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main detection loop with smooth 60 FPS display."""
        print("🎬 Starting smooth YOLO detection...")
        print("📋 Controls: 'q' to quit, 's' to save screenshot")
        print("🎯 Optimized for 60 FPS display with threaded detection")
        
        self.running = True
        
        # Start threads
        camera_thread = threading.Thread(target=self._camera_thread, daemon=True)
        detection_thread = threading.Thread(target=self._detection_thread, daemon=True)
        
        camera_thread.start()
        detection_thread.start()
        
        # Give threads time to start
        time.sleep(0.5)
        
        try:
            while True:
                try:
                    # Get latest frame (non-blocking)
                    frame = self.frame_queue.get_nowait()
                    
                    # Draw detections and info
                    annotated_frame = self._draw_detections(frame)
                    final_frame = self._draw_info_overlay(annotated_frame)
                    
                    # Display frame
                    cv2.imshow('Smooth YOLO Detection (60 FPS)', final_frame)
                    
                    # Handle keys
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"smooth_yolo_{int(time.time())}.jpg"
                        cv2.imwrite(filename, final_frame)
                        print(f"📸 Saved: {filename}")
                    
                except queue.Empty:
                    # No new frame available, skip this iteration
                    time.sleep(0.001)  # Small delay to prevent CPU spinning
                    continue
                
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            cv2.destroyAllWindows()
            
            # Wait for threads to finish
            camera_thread.join(timeout=1.0)
            detection_thread.join(timeout=1.0)
            
            print("✅ Smooth YOLO detection stopped")


def main():
    """Main function with enhanced argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Smooth YOLO Detection (60 FPS)')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model')
    parser.add_argument('--target', default=None, help='Target class (e.g., bottle, person)')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    try:
        detector = SmoothYOLODetector(args.model, args.target, args.confidence)
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your camera is working and not used by other apps")


if __name__ == "__main__":
    main() 