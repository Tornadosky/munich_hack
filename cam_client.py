"""
Camera client for object detection and UDP transmission.
Runs YOLOv8 detection on webcam feed and sends target detections to fusion server.
Supports automatic AprilTag-based pose calibration.
"""

import cv2
import json
import time
import argparse
import asyncio
import socket
import numpy as np
import math
from ultralytics import YOLO
from typing import Optional, Dict, Any, List


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


class AprilTagPoseEstimator:
    """AprilTag-based pose estimation for real-time camera calibration."""
    
    def __init__(self, camera_id: str, config_file: str = "apriltag_config.json"):
        """
        Initialize AprilTag pose estimator.
        
        Args:
            camera_id: Camera identifier
            config_file: Path to AprilTag configuration JSON file
        """
        self.camera_id = camera_id
        self.config_file = config_file
        
        # Load configuration
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"📋 Loaded AprilTag config from {config_file}")
        except FileNotFoundError:
            print(f"❌ AprilTag config file not found: {config_file}")
            print(f"   Please create the config file with your AprilTag positions")
            raise
        
        # Create tag position lookup
        self.tag_positions = {}
        for tag in self.config['apriltags']:
            tag_id = tag['id']
            self.tag_positions[tag_id] = {
                'position': np.array([tag['x'], tag['y'], tag['z']], dtype=np.float32),
                'size': tag.get('size', self.config['default_size']),
                'description': tag.get('description', f"Tag {tag_id}")
            }
        
        print(f"📍 Available AprilTags:")
        for tag_id, info in self.tag_positions.items():
            pos = info['position']
            print(f"   Tag {tag_id}: ({pos[0]}, {pos[1]}, {pos[2]}) - size: {info['size']}m")
        
        # Camera intrinsic parameters (will be updated when resolution is known)
        self.focal_length = self.config['camera_intrinsics']['focal_length']
        self.dist_coeffs = np.array(self.config['camera_intrinsics']['distortion_coeffs'], dtype=np.float32)
        self.camera_matrix = None
        
        # Initialize AprilTag detector with robotpy_apriltag
        try:
            import robotpy_apriltag as apriltag
            self.detector = apriltag.AprilTagDetector()
            
            # Configure the detector based on tag family
            if self.config['tag_family'] == 'tag36h11':
                self.detector.addFamily("tag36h11", 0)  # 0 bit correction
            elif self.config['tag_family'] == 'tag25h9':
                self.detector.addFamily("tag25h9", 0)
            elif self.config['tag_family'] == 'tag16h5':
                self.detector.addFamily("tag16h5", 0)
            else:
                # Default to tag36h11
                self.detector.addFamily("tag36h11", 0)
                print(f"⚠️  Unknown tag family '{self.config['tag_family']}', using tag36h11")
            
            print(f"✅ robotpy-apriltag detector ready with family {self.config['tag_family']}")
        except ImportError:
            print(f"❌ robotpy-apriltag library not found. Install with: pip install robotpy-apriltag")
            raise
        
        # Pose estimation parameters
        self.max_reproj_error = self.config['pose_estimation']['max_reproj_error']
        self.min_detection_confidence = self.config['pose_estimation']['min_detection_confidence']
        
        # Initialize detection statistics
        self.detection_stats = {
            'total_frames': 0,
            'frames_with_tags': 0,
            'total_tags_detected': 0,
            'successful_poses': 0,
            'failed_poses': 0,
            'last_detection_time': 0,
            'tag_detection_counts': {},  # Per-tag detection tracking
            'detection_rate_history': []  # Add missing field from working calibration
        }
        
        print(f"🔧 Detection confidence threshold: {self.min_detection_confidence}")
        print(f"🔧 Max reprojection error: {self.max_reproj_error}")
        print(f"🔧 AprilTag Pose Estimator initialized for camera {camera_id}")
    
    def set_camera_resolution(self, width: int, height: int):
        """Set camera resolution and update intrinsic matrix."""
        self.camera_matrix = np.array([
            [self.focal_length, 0, width / 2],
            [0, self.focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        print(f"📐 Camera matrix updated for resolution {width}x{height}")
        print(f"   Focal length: {self.focal_length}px")
        print(f"   Principal point: ({width/2}, {height/2})")
    
    def set_horizontal_flip(self, is_flipped: bool):
        """Set whether the camera feed is horizontally flipped."""
        self._is_horizontally_flipped = is_flipped
        print(f"🔄 AprilTag estimator: Horizontal flip {'ENABLED' if is_flipped else 'DISABLED'}")
        print(f"   This affects camera position calculation relative to AprilTags")
    
    def get_tag_object_points(self, tag_size: float) -> np.ndarray:
        """
        Get 3D object points for an AprilTag in tag coordinate system.
        
        Args:
            tag_size: Size of the tag in meters
            
        Returns:
            4x3 array of object points
        """
        half_size = tag_size / 2.0
        return np.array([
            [-half_size, -half_size, 0],  # Bottom-left
            [ half_size, -half_size, 0],  # Bottom-right
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0]   # Top-left
        ], dtype=np.float32)
    
    def detect_apriltags(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect AprilTags in frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected tag dictionaries
        """
        # Update detection statistics
        self.detection_stats['total_frames'] += 1
        current_time = time.time()
        
        # Convert to grayscale for AprilTag detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect tags using robotpy_apriltag
        start_time = time.time()
        detections = self.detector.detect(gray)
        detection_time = time.time() - start_time
        
        detected_tags = []
        rejected_tags = []
        
        for detection in detections:
            tag_id = detection.getId()
            confidence = detection.getDecisionMargin()
            
            # Log detection attempt
            if tag_id not in self.detection_stats['tag_detection_counts']:
                self.detection_stats['tag_detection_counts'][tag_id] = {'seen': 0, 'accepted': 0, 'rejected': 0}
            
            self.detection_stats['tag_detection_counts'][tag_id]['seen'] += 1
            
            # Check confidence threshold
            if confidence < self.min_detection_confidence:
                rejected_tags.append({'id': tag_id, 'confidence': confidence, 'reason': 'low_confidence'})
                self.detection_stats['tag_detection_counts'][tag_id]['rejected'] += 1
                continue
            
            # Check if tag is in our configuration
            if tag_id not in self.tag_positions:
                rejected_tags.append({'id': tag_id, 'confidence': confidence, 'reason': 'unknown_id'})
                print(f"⚠️  Detected unknown tag ID {tag_id} (confidence: {confidence:.3f}) - skipping")
                continue
            
            # Extract corner coordinates from robotpy_apriltag detection
            corners = np.array([
                [detection.getCorner(0).x, detection.getCorner(0).y],
                [detection.getCorner(1).x, detection.getCorner(1).y],
                [detection.getCorner(2).x, detection.getCorner(2).y],
                [detection.getCorner(3).x, detection.getCorner(3).y]
            ], dtype=np.float32)
            
            # Calculate center point
            center = detection.getCenter()
            center_point = (center.x, center.y)
            
            # Calculate tag area and aspect ratio for quality assessment
            tag_area = cv2.contourArea(corners)
            x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
            y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / height if height > 0 else 0
            
            detected_tags.append({
                'id': tag_id,
                'corners': corners,
                'center': center_point,
                'confidence': confidence,
                'tag_info': self.tag_positions[tag_id],
                'area': tag_area,
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'detection_time': detection_time
            })
            
            self.detection_stats['tag_detection_counts'][tag_id]['accepted'] += 1
        
        # Update statistics
        if detected_tags:
            self.detection_stats['frames_with_tags'] += 1
            self.detection_stats['total_tags_detected'] += len(detected_tags)
            self.detection_stats['last_detection_time'] = current_time
        
        # Log detection summary periodically (every 30 frames to match working calibration)
        if self.detection_stats['total_frames'] % 30 == 0:  # Every 30 frames like working calibration
            self._log_detection_status(detected_tags, rejected_tags, detection_time)
        
        return detected_tags
    
    def _log_detection_status(self, detected_tags, rejected_tags, detection_time):
        """Log detailed detection status."""
        stats = self.detection_stats
        detection_rate = (stats['frames_with_tags'] / stats['total_frames']) * 100 if stats['total_frames'] > 0 else 0
        
        print(f"\n📊 APRILTAG DETECTION STATUS for Camera {self.camera_id}:")
        print(f"   Frames processed: {stats['total_frames']}")
        print(f"   Detection rate: {detection_rate:.1f}% ({stats['frames_with_tags']}/{stats['total_frames']})")
        print(f"   Total tags detected: {stats['total_tags_detected']}")
        print(f"   Detection time: {detection_time*1000:.1f}ms")
        
        if detected_tags:
            print(f"   Current frame: {len(detected_tags)} tags detected")
            for tag in detected_tags:
                tag_id = tag['id']
                pos = self.tag_positions[tag_id]['position']
                print(f"      Tag {tag_id}: conf={tag['confidence']:.3f}, area={tag['area']:.0f}px², "
                      f"aspect={tag['aspect_ratio']:.2f}, pos=({pos[0]}, {pos[1]})")
        
        if rejected_tags:
            print(f"   Rejected tags: {len(rejected_tags)}")
            for tag in rejected_tags:
                print(f"      Tag {tag['id']}: conf={tag['confidence']:.3f}, reason={tag['reason']}")
        
        # Per-tag statistics
        if stats['tag_detection_counts']:
            print(f"   Tag detection history:")
            for tag_id, counts in stats['tag_detection_counts'].items():
                success_rate = (counts['accepted'] / counts['seen']) * 100 if counts['seen'] > 0 else 0
                print(f"      Tag {tag_id}: {counts['accepted']}/{counts['seen']} ({success_rate:.1f}% success)")
        else:
            print(f"   No tags detected yet")
        
        print("=" * 50)
    
    def estimate_pose_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Estimate camera pose from AprilTags in frame.
        
        Args:
            frame: Camera frame
            
        Returns:
            Pose dictionary or None if no tags detected
        """
        print(f"🔍 DEBUG estimate_pose_from_frame: camera_matrix is None: {self.camera_matrix is None}")
        if self.camera_matrix is None:
            print(f"❌ DEBUG: Camera matrix is None, cannot estimate pose")
            self._last_frame_tags = []  # Store empty tags for display
            return None
        
        # Detect AprilTags
        print(f"🔍 DEBUG: Calling detect_apriltags...")
        detected_tags = self.detect_apriltags(frame)
        print(f"🔍 DEBUG: Detected {len(detected_tags)} tags: {[tag['id'] for tag in detected_tags]}")
        
        # Store detected tags for display reuse
        self._last_frame_tags = detected_tags
        
        if not detected_tags:
            print(f"❌ DEBUG: No tags detected, returning None")
            return None
        
        # Use the first detected tag for pose estimation
        tag = detected_tags[0]
        tag_id = tag['id']
        corners = tag['corners']
        tag_info = tag['tag_info']
        
        print(f"🔍 DEBUG: Using tag {tag_id} for pose estimation")
        print(f"🔍 DEBUG: Tag corners shape: {corners.shape}")
        print(f"🔍 DEBUG: Tag world position: {tag_info['position']}")
        print(f"🔍 DEBUG: Tag size: {tag_info['size']}m")
        
        # Get tag size and world position
        tag_size = tag_info['size']
        tag_world_pos = tag_info['position']
        
        # Get 3D object points in tag coordinate system
        tag_object_points = self.get_tag_object_points(tag_size)
        print(f"🔍 DEBUG: Tag object points shape: {tag_object_points.shape}")
        
        try:
            print(f"🔍 DEBUG: Calling solvePnP...")
            # Solve PnP to get tag pose relative to camera
            success, rvec, tvec = cv2.solvePnP(
                tag_object_points,
                corners,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            print(f"🔍 DEBUG: solvePnP success: {success}")
            if not success:
                print(f"❌ DEBUG: solvePnP failed")
                return None
            
            print(f"🔍 DEBUG: rvec: {rvec.flatten()}")
            print(f"🔍 DEBUG: tvec: {tvec.flatten()}")
            
            # Convert rotation vector to rotation matrix
            R_tag_to_cam, _ = cv2.Rodrigues(rvec)
            t_tag_to_cam = tvec.flatten()
            
            # Transform from tag coordinate system to world coordinate system
            # Camera position in tag coordinates
            cam_pos_in_tag = -R_tag_to_cam.T @ t_tag_to_cam
            print(f"🔍 DEBUG: cam_pos_in_tag: {cam_pos_in_tag}")
            
            # IMPORTANT: Account for horizontal flip in camera coordinate system
            # If the frame is horizontally flipped, we need to mirror the X coordinate
            # This is because the AprilTag detection was done on the original frame,
            # but the camera's actual viewing direction is mirrored
            if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                # Mirror the X component of camera position relative to tag
                cam_pos_in_tag[0] = -cam_pos_in_tag[0]
                print(f"🔍 DEBUG: cam_pos_in_tag after flip correction: {cam_pos_in_tag}")
            
            # Camera position in world coordinates
            # For now, assume tag is axis-aligned (no rotation)
            camera_world_pos = tag_world_pos + cam_pos_in_tag
            print(f"🔍 DEBUG: camera_world_pos: {camera_world_pos}")
            
            # Calculate camera orientation
            # Camera Z-axis in tag coordinates (pointing toward tag)
            cam_z_in_tag = R_tag_to_cam.T @ np.array([0, 0, 1])
            
            # Account for horizontal flip in orientation calculation
            if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                # Mirror the X component of camera direction
                cam_z_in_tag[0] = -cam_z_in_tag[0]
                print(f"🔍 DEBUG: cam_z_in_tag after flip correction: {cam_z_in_tag}")
            
            # Project to XY plane and calculate yaw
            yaw_rad = math.atan2(-cam_z_in_tag[1], -cam_z_in_tag[0])
            yaw_deg = math.degrees(yaw_rad)
            print(f"🔍 DEBUG: yaw_deg: {yaw_deg}")
            
            # Calculate distance to tag
            distance_to_tag = float(np.linalg.norm(t_tag_to_cam))
            print(f"🔍 DEBUG: distance_to_tag: {distance_to_tag}")
            
            # Estimate FOV based on tag size in image
            tag_pixel_width = np.max(corners[:, 0]) - np.min(corners[:, 0])
            
            if distance_to_tag > 0 and tag_pixel_width > 0:
                tag_angular_width = 2 * math.atan(tag_size / (2 * distance_to_tag))
                estimated_fov = math.degrees(tag_angular_width) * (frame.shape[1] / tag_pixel_width)
                estimated_fov = max(30.0, min(120.0, estimated_fov))
            else:
                estimated_fov = 70.0
            
            pose = {
                'cam_id': self.camera_id,
                'x': float(camera_world_pos[0]),
                'y': float(camera_world_pos[1]),
                'z': float(camera_world_pos[2]),
                'yaw_deg': float(yaw_deg),
                'fov_deg': float(estimated_fov),
                'img_w': frame.shape[1],
                'img_h': frame.shape[0],
                'distance_to_tag': distance_to_tag,
                'reference_tag_id': tag_id,
                'reference_tag_pos': tag_world_pos.tolist(),
                'calibration_method': 'apriltag_realtime',
                'timestamp': time.time()
            }
            
            print(f"✅ DEBUG: Successfully calculated pose: {pose}")
            return pose
            
        except Exception as e:
            print(f"❌ DEBUG: AprilTag pose estimation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of detection statistics."""
        stats = self.detection_stats
        detection_rate = (stats['frames_with_tags'] / stats['total_frames']) * 100 if stats['total_frames'] > 0 else 0
        pose_success_rate = (stats['successful_poses'] / (stats['successful_poses'] + stats['failed_poses'])) * 100 if (stats['successful_poses'] + stats['failed_poses']) > 0 else 0
        
        return {
            'total_frames': stats['total_frames'],
            'detection_rate_percent': detection_rate,
            'total_tags_detected': stats['total_tags_detected'],
            'successful_poses': stats['successful_poses'],
            'failed_poses': stats['failed_poses'],
            'pose_success_rate_percent': pose_success_rate,
            'tag_detection_counts': stats['tag_detection_counts'].copy(),
            'last_detection_ago_seconds': time.time() - stats['last_detection_time'] if stats['last_detection_time'] > 0 else None
        }


class CameraClient:
    """Webcam object detection client with UDP transmission and AprilTag pose estimation."""
    
    def __init__(self, pose_file: str, server_addr: str, target_class: str, fps: float, 
                 flip_horizontal: bool = True, enable_apriltag_pose: bool = False, 
                 apriltag_config: str = "apriltag_config.json", apriltag_update_interval: float = 1.0):
        """
        Initialize camera client with configuration.
        
        Args:
            pose_file: Path to camera pose JSON file
            server_addr: Server address as "host:port"
            target_class: YOLO class name to detect (e.g., "bottle")
            fps: Detection frame rate
            flip_horizontal: Whether to flip the camera feed horizontally (default True)
            enable_apriltag_pose: Enable AprilTag-based pose estimation
            apriltag_config: AprilTag configuration file
            apriltag_update_interval: AprilTag pose update interval in seconds
        """
        # Load camera pose configuration
        with open(pose_file, 'r') as f:
            self.pose = json.load(f)
        
        # Store initial pose as fallback
        self.initial_pose = self.pose.copy()
        
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
        self.enable_apriltag_pose = enable_apriltag_pose
        
        # AprilTag pose estimation
        self.apriltag_estimator = None
        if enable_apriltag_pose:
            try:
                self.apriltag_estimator = AprilTagPoseEstimator(self.pose['cam_id'], apriltag_config)
                print(f"🏷️  AprilTag pose estimation enabled (config: {apriltag_config}, update interval: {apriltag_update_interval}s)")
            except Exception as e:
                print(f"❌ Failed to initialize AprilTag estimator: {e}")
                self.enable_apriltag_pose = False
        
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
        
        # AprilTag pose tracking
        self.last_apriltag_update = 0
        self.apriltag_update_interval = apriltag_update_interval
        self.apriltag_update_count = 0  # Track number of updates
        
        # Camera pose broadcast (send pose info even without detections)
        self.last_pose_broadcast = 0
        self.pose_broadcast_interval = 0.5  # Send pose every 0.5 seconds even without detections
        self.pose_broadcast_count = 0
    
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
    
    def update_pose_from_apriltags(self, frame: np.ndarray) -> bool:
        """
        Update camera pose using AprilTag detection.
        
        Args:
            frame: Current camera frame
            
        Returns:
            True if pose was updated, False otherwise
        """
        if not self.enable_apriltag_pose or self.apriltag_estimator is None:
            return False
        
        current_time = time.time()
        if current_time - self.last_apriltag_update < self.apriltag_update_interval:
            return False
        
        # DEBUG: Always try pose estimation to see what's happening
        print(f"\n🔍 DEBUG: Attempting AprilTag pose estimation...")
        
        # Try to estimate pose from AprilTags
        apriltag_pose = self.apriltag_estimator.estimate_pose_from_frame(frame)
        
        if apriltag_pose is not None:
            print(f"✅ DEBUG: Got AprilTag pose: {apriltag_pose}")
            
            # Update current pose with AprilTag-based pose
            old_pose = self.pose.copy()
            
            self.pose.update({
                'x': apriltag_pose['x'],
                'y': apriltag_pose['y'], 
                'yaw_deg': apriltag_pose['yaw_deg'],
                'fov_deg': apriltag_pose['fov_deg']
            })
            
            self.last_apriltag_update = current_time
            self.apriltag_update_count += 1
            
            # Calculate position change
            position_change = math.sqrt((apriltag_pose['x'] - old_pose['x'])**2 + (apriltag_pose['y'] - old_pose['y'])**2)
            yaw_change = abs(apriltag_pose['yaw_deg'] - old_pose['yaw_deg'])
            if yaw_change > 180:
                yaw_change = 360 - yaw_change  # Handle wraparound
            
            # Get detection statistics summary
            detection_summary = self.apriltag_estimator.get_detection_summary()
            
            print(f"\n🏷️  APRILTAG POSE UPDATE #{self.apriltag_update_count} for Camera {self.pose['cam_id']}:")
            print(f"   📍 CAMERA POSITION:")
            print(f"      Old: ({old_pose['x']:.3f}, {old_pose['y']:.3f}) m")
            print(f"      New: ({self.pose['x']:.3f}, {self.pose['y']:.3f}) m")
            print(f"      Change: {position_change:.3f}m")
            print(f"   🧭 CAMERA ORIENTATION:")
            print(f"      Old: {old_pose['yaw_deg']:.1f}°")
            print(f"      New: {self.pose['yaw_deg']:.1f}°")
            print(f"      Change: {yaw_change:.1f}°")
            print(f"   🔄 COORDINATE SYSTEM:")
            print(f"      Horizontal flip: {'ENABLED' if self.flip_horizontal else 'DISABLED'}")
            print(f"      Flip correction applied: {'YES' if hasattr(self.apriltag_estimator, '_is_horizontally_flipped') and self.apriltag_estimator._is_horizontally_flipped else 'NO'}")
            print(f"   🏷️  APRILTAG DETECTION:")
            print(f"      Reference Tag ID: {apriltag_pose['reference_tag_id']}")
            print(f"      Tag Position: {apriltag_pose['reference_tag_pos']}")
            print(f"      Distance to Tag: {apriltag_pose['distance_to_tag']:.3f}m")
            print(f"      Detection Confidence: {apriltag_pose.get('confidence', 'N/A')}")
            print(f"      Reprojection Error: {apriltag_pose.get('reprojection_error', 0):.2f}px")
            print(f"      Estimated FOV: {apriltag_pose['fov_deg']:.1f}°")
            print(f"   📊 DETECTION STATISTICS:")
            print(f"      Detection Rate: {detection_summary['detection_rate_percent']:.1f}%")
            print(f"      Pose Success Rate: {detection_summary['pose_success_rate_percent']:.1f}%")
            print(f"      Total Tags Detected: {detection_summary['total_tags_detected']}")
            print(f"      Last Detection: {detection_summary.get('last_detection_ago_seconds', 'N/A')} seconds ago")
            print(f"   📡 SERVER TRANSMISSION:")
            print(f"      Updated pose will be sent with next detection packet")
            print(f"      Server: {self.server_addr[0]}:{self.server_addr[1]}")
            
            # Send immediate pose broadcast to update server with new position
            self.send_immediate_pose_broadcast()
            
            return True
        else:
            print(f"❌ DEBUG: No AprilTag pose detected")
            
            # Log detection failure periodically
            if self.apriltag_update_count == 0 or (current_time - self.last_apriltag_update) > 10.0:  # Every 10 seconds if no updates
                detection_summary = self.apriltag_estimator.get_detection_summary()
                print(f"\n⚠️  AprilTag pose update failed for Camera {self.pose['cam_id']}:")
                print(f"   📊 Detection Statistics:")
                print(f"      Frames processed: {detection_summary['total_frames']}")
                print(f"      Detection rate: {detection_summary['detection_rate_percent']:.1f}%")
                print(f"      Successful poses: {detection_summary['successful_poses']}")
                print(f"      Failed poses: {detection_summary['failed_poses']}")
                
                if detection_summary['tag_detection_counts']:
                    print(f"   🏷️  Tag Detection Status:")
                    for tag_id, counts in detection_summary['tag_detection_counts'].items():
                        success_rate = (counts['accepted'] / counts['seen']) * 100 if counts['seen'] > 0 else 0
                        print(f"      Tag {tag_id}: {counts['accepted']}/{counts['seen']} ({success_rate:.1f}% success)")
                else:
                    print(f"   🏷️  No AprilTags detected yet")
                
                # Update last update time to prevent spam
                self.last_apriltag_update = current_time
        
        return False
    
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
    
    def send_immediate_pose_broadcast(self) -> bool:
        """
        Send camera pose information immediately (used when pose is updated).
        This bypasses the normal broadcast interval.
        
        Returns:
            True if packet was sent successfully
        """
        current_time = time.time()
        
        packet = {
            "cam_id": self.pose['cam_id'],
            "pose_broadcast": True,
            "immediate_update": True,  # Flag to indicate this is an immediate pose update
            "timestamp": current_time,
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
            self.last_pose_broadcast = current_time  # Update broadcast time to avoid spam
            self.pose_broadcast_count += 1
            print(f"📡 Sent IMMEDIATE pose broadcast #{self.pose_broadcast_count} for camera {self.pose['cam_id']} at ({self.pose['x']:.3f}, {self.pose['y']:.3f})")
        else:
            print(f"❌ Failed to send immediate pose broadcast for camera {self.pose['cam_id']}")
        
        return success
    
    def send_pose_broadcast(self) -> bool:
        """
        Send camera pose information to server even without detections.
        This ensures the camera appears in the fusion server plot.
        
        Returns:
            True if packet was sent successfully
        """
        current_time = time.time()
        if current_time - self.last_pose_broadcast < self.pose_broadcast_interval:
            return False
        
        packet = {
            "cam_id": self.pose['cam_id'],
            "pose_broadcast": True,  # Flag to indicate this is a pose-only packet
            "timestamp": current_time,
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
            self.last_pose_broadcast = current_time
            self.pose_broadcast_count += 1
            print(f"📡 Sent pose broadcast #{self.pose_broadcast_count} for camera {self.pose['cam_id']}")
        else:
            print(f"❌ Failed to send pose broadcast for camera {self.pose['cam_id']}")
        
        return success
    
    async def run_detection_loop(self):
        """Main detection loop running at specified FPS."""
        self.start_capture()
        
        # Initialize AprilTag estimator with actual resolution
        if self.apriltag_estimator is not None:
            self.apriltag_estimator.set_camera_resolution(self.actual_resolution[0], self.actual_resolution[1])
            # Set horizontal flip flag so pose calculation accounts for mirrored coordinates
            self.apriltag_estimator.set_horizontal_flip(self.flip_horizontal)
            print(f"🏷️  AprilTag estimator ready with resolution {self.actual_resolution[0]}x{self.actual_resolution[1]}")
        
        frame_count = 0
        detection_count = 0
        successful_sends = 0
        failed_sends = 0
        apriltag_pose_updates = 0
        
        try:
            print(f"🚀 Starting detection loop at {1/self.detection_interval:.1f} FPS")
            print(f"🎯 Looking for '{self.target_class}' objects")
            if self.enable_apriltag_pose:
                print(f"🏷️  AprilTag pose estimation enabled - camera pose will auto-update")
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
                
                # Store original frame for AprilTag detection (before flip)
                original_frame = frame.copy()
                
                # Try to update pose from AprilTags using ORIGINAL frame (before flip)
                apriltag_updated = self.update_pose_from_apriltags(original_frame)
                if apriltag_updated:
                    apriltag_pose_updates += 1
                
                # Apply horizontal flip AFTER AprilTag detection for display and YOLO
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)  # 1 = horizontal flip
                
                # Get AprilTag detection results for display (reuse from pose update if available)
                detected_tags = []
                if self.enable_apriltag_pose and self.apriltag_estimator is not None:
                    # Get the most recent detection results without re-detecting
                    detected_tags = getattr(self.apriltag_estimator, '_last_frame_tags', [])
                
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
                res_text = f"Resolution: {self.actual_resolution[0]}x{self.actual_resolution[1]} | Flip: {'ON' if self.flip_horizontal else 'Disabled'}"
                cv2.putText(display_frame, res_text,
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show AprilTag pose status if enabled - COMPREHENSIVE DEBUG INFO
                if self.enable_apriltag_pose and self.apriltag_estimator is not None:
                    y_offset = 120
                    
                    # Show AprilTag detection statistics
                    apriltag_text = f"AprilTag Updates: {self.apriltag_update_count} | Last: {time.time() - self.last_apriltag_update:.1f}s ago"
                    cv2.putText(display_frame, apriltag_text,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 30
                    
                    # Show detection rate and frame processing
                    detect_rate_text = f"AprilTag Rate: {self.apriltag_estimator.get_detection_summary()['detection_rate_percent']:.1f}% | Frames: {self.apriltag_estimator.get_detection_summary()['total_frames']}"
                    cv2.putText(display_frame, detect_rate_text,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 25
                    
                    # Show pose success rate
                    pose_success_text = f"Pose Success: {self.apriltag_estimator.get_detection_summary()['pose_success_rate_percent']:.1f}% | Total Tags: {self.apriltag_estimator.get_detection_summary()['total_tags_detected']}"
                    cv2.putText(display_frame, pose_success_text,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 25
                    
                    # Show current pose
                    pose_text = f"Pose: ({self.pose['x']:.2f}, {self.pose['y']:.2f}) | Yaw: {self.pose['yaw_deg']:.1f}° | FOV: {self.pose['fov_deg']:.1f}°"
                    cv2.putText(display_frame, pose_text,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 25
                    
                    # Show detected tags in current frame with detailed info
                    if detected_tags:
                        tags_text = f"Current Frame: {len(detected_tags)} tags detected"
                        cv2.putText(display_frame, tags_text,
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20
                        
                        # Show each detected tag
                        for i, tag in enumerate(detected_tags[:3]):  # Show max 3 tags to avoid clutter
                            tag_id = tag['id']
                            confidence = tag['confidence']
                            area = tag.get('area', 0)
                            pos = tag['tag_info']['position']
                            
                            tag_detail = f"  Tag {tag_id}: conf={confidence:.3f}, area={area:.0f}px², pos=({pos[0]}, {pos[1]})"
                            cv2.putText(display_frame, tag_detail,
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            y_offset += 18
                            
                            # Draw tag detection on frame
                            corners = tag['corners']
                            center = tag['center']
                            
                            # Transform coordinates for display on flipped frame
                            display_corners = self.transform_apriltag_coords_for_display(corners, display_frame.shape[1])
                            display_center = self.transform_apriltag_coords_for_display(center, display_frame.shape[1])
                            
                            # Debug: Show coordinate transformation
                            if frame_count % 30 == 0:  # Every 30 frames to avoid spam
                                print(f"🔄 DEBUG Tag {tag_id} coordinate transform:")
                                print(f"   Original center: ({center[0]:.1f}, {center[1]:.1f})")
                                print(f"   Display center: ({display_center[0]:.1f}, {display_center[1]:.1f})")
                                print(f"   Frame width: {display_frame.shape[1]}, Flip enabled: {self.flip_horizontal}")
                            
                            # Draw tag corners and outline
                            corners_int = display_corners.astype(int)
                            cv2.polylines(display_frame, [corners_int], True, (0, 255, 0), 2)
                            
                            # Draw tag ID and confidence at center
                            cv2.putText(display_frame, f"ID:{tag_id}", 
                                       (int(display_center[0]) - 30, int(display_center[1]) - 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"C:{confidence:.2f}", 
                                       (int(display_center[0]) - 30, int(display_center[1])),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            # Draw center point
                            cv2.circle(display_frame, (int(display_center[0]), int(display_center[1])), 5, (0, 0, 255), -1)
                    else:
                        no_tags_text = "Current Frame: No AprilTags detected"
                        cv2.putText(display_frame, no_tags_text,
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        y_offset += 25
                    
                    # Show per-tag detection history
                    if self.apriltag_estimator.get_detection_summary()['tag_detection_counts']:
                        history_text = "Tag History:"
                        cv2.putText(display_frame, history_text,
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 20
                        
                        for tag_id, counts in self.apriltag_estimator.get_detection_summary()['tag_detection_counts'].items():
                            success_rate = (counts['accepted'] / counts['seen']) * 100 if counts['seen'] > 0 else 0
                            tag_history = f"  Tag {tag_id}: {counts['accepted']}/{counts['seen']} ({success_rate:.1f}%)"
                            cv2.putText(display_frame, tag_history,
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            y_offset += 18
                    
                    # Show server connection status
                    server_text = f"Server: {self.server_addr[0]}:{self.server_addr[1]} | Updates: {self.apriltag_update_count} | Broadcasts: {self.pose_broadcast_count}"
                    cv2.putText(display_frame, server_text,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
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
                    status_msg = f"📊 Status: {frame_count} frames, {detection_count} detections, {successful_sends} sent, {failed_sends} failed"
                    if self.enable_apriltag_pose:
                        if hasattr(self.apriltag_estimator, 'get_detection_summary'):
                            summary = self.apriltag_estimator.get_detection_summary()
                            status_msg += f", {apriltag_pose_updates} AprilTag updates ({summary['detection_rate_percent']:.1f}% detection rate)"
                        else:
                            status_msg += f", {apriltag_pose_updates} AprilTag updates"
                    print(status_msg)
                
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
            if self.enable_apriltag_pose:
                print(f"   AprilTag pose updates: {apriltag_pose_updates}")
                if hasattr(self.apriltag_estimator, 'get_detection_summary'):
                    summary = self.apriltag_estimator.get_detection_summary()
                    print(f"   AprilTag detection rate: {summary['detection_rate_percent']:.1f}%")
                    print(f"   AprilTag pose success rate: {summary['pose_success_rate_percent']:.1f}%")
                    print(f"   Total AprilTags detected: {summary['total_tags_detected']}")
            print(f"   Pose broadcasts sent: {self.pose_broadcast_count}")
            if frame_count > 0:
                print(f"   Detection rate: {detection_count/frame_count:.2f} detections/frame")
            
            self.cap.release()
            cv2.destroyAllWindows()

    def transform_apriltag_coords_for_display(self, coords, frame_width):
        """
        Transform AprilTag coordinates from original frame to flipped display frame.
        
        Args:
            coords: Coordinates from original frame (before flip)
            frame_width: Width of the frame
            
        Returns:
            Transformed coordinates for display on flipped frame
        """
        if not self.flip_horizontal:
            return coords
        
        # For horizontal flip, mirror X coordinates
        if isinstance(coords, tuple):
            # Single point (x, y)
            return (frame_width - coords[0], coords[1])
        elif isinstance(coords, np.ndarray) and coords.shape == (4, 2):
            # AprilTag corners array
            transformed = coords.copy()
            transformed[:, 0] = frame_width - coords[:, 0]
            return transformed
        else:
            # Generic array of points
            transformed = coords.copy()
            if len(transformed.shape) == 2 and transformed.shape[1] >= 1:
                transformed[:, 0] = frame_width - coords[:, 0]
            return transformed


async def main():
    """Parse arguments and run camera client."""
    parser = argparse.ArgumentParser(description='Camera client for object detection with optional AprilTag pose estimation')
    parser.add_argument('--pose', required=True, help='Camera pose JSON file')
    parser.add_argument('--server', required=True, help='Server address (host:port)')
    parser.add_argument('--target', default='bottle', help='Target object class')
    parser.add_argument('--fps', type=float, default=2.0, help='Detection FPS')
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flip (enabled by default)')
    parser.add_argument('--enable-apriltag-pose', action='store_true', 
                       help='Enable AprilTag-based pose estimation (requires AprilTag config file)')
    parser.add_argument('--apriltag-config', default='apriltag_config.json',
                       help='AprilTag configuration file (default: apriltag_config.json)')
    parser.add_argument('--apriltag-update-interval', type=float, default=1.0,
                       help='AprilTag pose update interval in seconds (default: 1.0s)')
    
    args = parser.parse_args()
    
    # Flip is enabled by default, disabled only if --no-flip is specified
    flip_horizontal = not args.no_flip
    
    if args.enable_apriltag_pose:
        print(f"🏷️  AprilTag pose estimation enabled:")
        print(f"   Config file: {args.apriltag_config}")
        print(f"   Update interval: {args.apriltag_update_interval}s")
        print(f"   💡 Place AprilTags at the positions specified in {args.apriltag_config}")
        print(f"   💡 Camera pose will automatically update every {args.apriltag_update_interval}s when tags are detected")
    
    # Create and run client
    try:
        client = CameraClient(
            args.pose, 
            args.server, 
            args.target, 
            args.fps, 
            flip_horizontal,
            args.enable_apriltag_pose,
            args.apriltag_config,
            args.apriltag_update_interval
        )
        await client.run_detection_loop()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 