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
                'description': tag.get('description', f"Tag {tag_id}"),
                'orientation': tag.get('orientation', 'horizontal'),
                'wall_normal': tag.get('wall_normal', [0, 1, 0]),
                'wall_direction': tag.get('wall_direction', 'north')
            }
        
        print(f"📍 Available AprilTags:")
        for tag_id, info in self.tag_positions.items():
            pos = info['position']
            orientation = info['orientation']
            print(f"   Tag {tag_id}: ({pos[0]}, {pos[1]}, {pos[2]}) - size: {info['size']}m - orientation: {orientation.upper()}")
        
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
        
        # Distance scaling for visualization (default 1.0, can be overridden)
        self.distance_scale_factor = 1.0
        
        # Verbose debugging flag (can be enabled for troubleshooting)
        self.verbose_debug = False
        
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
    
    def get_tag_object_points(self, tag_size: float, tag_info: dict) -> np.ndarray:
        """
        Get 3D object points for an AprilTag in tag coordinate system.
        
        Args:
            tag_size: Size of the tag in meters
            tag_info: Tag information including orientation
            
        Returns:
            4x3 array of object points
        """
        half_size = tag_size / 2.0
        
        # Get tag orientation
        orientation = tag_info.get('orientation', 'horizontal')
        
        if orientation == 'wall_mounted':
            # Wall-mounted tag - simplified for 2D positioning
            # Tag is vertical on wall, facing outward (toward cameras)
            return np.array([
                [-half_size, 0, -half_size],  # Bottom-left
                [ half_size, 0, -half_size],  # Bottom-right
                [ half_size, 0,  half_size],  # Top-right
                [-half_size, 0,  half_size]   # Top-left
            ], dtype=np.float32)
        else:
            # Horizontal tag (on floor) - original
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
        
        if self.verbose_debug:
            print(f"🔍 FRAME DEBUG: Processing frame #{self.detection_stats['total_frames']}")
            print(f"   Frame shape: {frame.shape}")
            print(f"   Frame dtype: {frame.dtype}")
            print(f"   Frame min/max values: {frame.min()}/{frame.max()}")
        
        # Convert to grayscale for AprilTag detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.verbose_debug:
                print(f"   Converted to grayscale: {gray.shape}")
        else:
            gray = frame
            if self.verbose_debug:
                print(f"   Already grayscale: {gray.shape}")
        
        if self.verbose_debug:
            print(f"   Grayscale min/max: {gray.min()}/{gray.max()}")
        
        # Detect tags using robotpy_apriltag
        start_time = time.time()
        if self.verbose_debug:
            print(f"   Calling detector.detect() with robotpy_apriltag...")
        try:
            detections = self.detector.detect(gray)
            detection_time = time.time() - start_time
            if self.verbose_debug:
                print(f"   ✅ Detection completed in {detection_time*1000:.1f}ms")
            print(f"   Raw detections found: {len(detections)}")
        except Exception as e:
            print(f"   ❌ Detection failed with error: {e}")
            return []
        
        detected_tags = []
        rejected_tags = []
        
        for i, detection in enumerate(detections):
            try:
                tag_id = detection.getId()
                confidence = detection.getDecisionMargin()
                
                if self.verbose_debug:
                    print(f"   Processing detection {i+1}: Tag ID {tag_id}, confidence {confidence:.3f}")
                
                # Log detection attempt
                if tag_id not in self.detection_stats['tag_detection_counts']:
                    self.detection_stats['tag_detection_counts'][tag_id] = {'seen': 0, 'accepted': 0, 'rejected': 0}
                
                self.detection_stats['tag_detection_counts'][tag_id]['seen'] += 1
                
                # Check confidence threshold
                if confidence < self.min_detection_confidence:
                    rejected_tags.append({'id': tag_id, 'confidence': confidence, 'reason': 'low_confidence'})
                    self.detection_stats['tag_detection_counts'][tag_id]['rejected'] += 1
                    if self.verbose_debug:
                        print(f"      ❌ Rejected: confidence {confidence:.3f} < threshold {self.min_detection_confidence}")
                    continue
                
                # Check if tag is in our configuration
                if tag_id not in self.tag_positions:
                    rejected_tags.append({'id': tag_id, 'confidence': confidence, 'reason': 'unknown_id'})
                    if self.verbose_debug:
                        print(f"      ❌ Rejected: unknown tag ID {tag_id} (not in config)")
                    else:
                        print(f"⚠️  Detected unknown tag ID {tag_id} (confidence: {confidence:.3f}) - not in config")
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
                
                if self.verbose_debug:
                    print(f"      ✅ Accepted: center=({center_point[0]:.1f}, {center_point[1]:.1f}), area={tag_area:.0f}px², aspect={aspect_ratio:.2f}")
                
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
                
            except Exception as e:
                print(f"      ❌ Error processing detection {i+1}: {e}")
                continue
        
        # Update statistics
        if detected_tags:
            self.detection_stats['frames_with_tags'] += 1
            self.detection_stats['total_tags_detected'] += len(detected_tags)
            self.detection_stats['last_detection_time'] = current_time
        
        # Always show detection result summary (not just in verbose mode)
        if detected_tags or rejected_tags:
            print(f"   📊 FRAME #{self.detection_stats['total_frames']}: {len(detected_tags)} accepted, {len(rejected_tags)} rejected")
            if detected_tags:
                for tag in detected_tags:
                    print(f"      ✅ Tag {tag['id']}: confidence={tag['confidence']:.3f}, area={tag['area']:.0f}px²")
            if rejected_tags:
                for tag in rejected_tags:
                    print(f"      ❌ Tag {tag['id']}: confidence={tag['confidence']:.3f}, reason={tag['reason']}")
        
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
        
        # Get tag orientation
        orientation = tag_info.get('orientation', 'horizontal')
        print(f"🔍 DEBUG: Tag orientation: {orientation.upper()}")
        
        if orientation == 'wall_mounted':
            print(f"🔍 DEBUG: Using simplified 2D positioning for wall-mounted tag")
            
            # Use simplified 2D positioning for wall-mounted tags
            pos_2d = self.estimate_2d_camera_position(tag_info, corners, frame.shape)
            if pos_2d is None:
                print(f"❌ DEBUG: 2D positioning failed")
                return None
            
            # Create pose result
            pose = {
                'cam_id': self.camera_id,
                'x': float(pos_2d['x']),
                'y': float(pos_2d['y']),
                'z': float(pos_2d['z']),
                'yaw_deg': float(pos_2d['yaw_deg']),
                'fov_deg': 70.0,  # Default FOV
                'img_w': frame.shape[1],
                'img_h': frame.shape[0],
                'distance_to_tag': float(pos_2d['distance_to_tag']),
                'reference_tag_id': tag_id,
                'reference_tag_pos': tag_info['position'].tolist(),
                'reference_tag_orientation': orientation.upper(),
                'calibration_method': 'apriltag_2d_simplified',
                'horizontal_offset': float(pos_2d['horizontal_offset']),
                'timestamp': time.time()
            }
            
            print(f"✅ DEBUG: 2D positioning successful: {pose}")
            return pose
            
        else:
            print(f"🔍 DEBUG: Using 3D solvePnP for {orientation} tag")
        
        # Continue with original 3D approach for non-wall-mounted tags
        if orientation == 'vertical':
            print(f"🔍 DEBUG: Vertical tag mounted on wall at position {tag_info['position']}")
        else:
            print(f"🔍 DEBUG: Horizontal tag lying flat on floor at Z={tag_info['position'][2]}")
        
        # Get tag size and world position
        tag_size = tag_info['size']
        tag_world_pos = tag_info['position']
        
        # Get 3D object points in tag coordinate system
        tag_object_points = self.get_tag_object_points(tag_size, tag_info)
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
            # Handle both VERTICAL tags (on walls) and HORIZONTAL tags (on floor)
            
            # Camera position in tag coordinates
            cam_pos_in_tag = -R_tag_to_cam.T @ t_tag_to_cam
            print(f"🔍 DEBUG: cam_pos_in_tag (raw): {cam_pos_in_tag}")
            
            # Get tag orientation
            orientation = tag_info.get('orientation', 'horizontal')
            
            if orientation == 'vertical':
                # VERTICAL TAG (mounted on wall)
                # Tag coordinate system: X=horizontal, Y=depth from wall, Z=vertical
                # World coordinate system: X=world X, Y=world Y, Z=height
                
                # For vertical tags, we need to transform based on which wall they're on
                normal_vector = tag_info.get('normal_vector', [0, 1, 0])  # Default facing +Y
                rotation_z = tag_info.get('rotation_z', 0.0)  # Rotation around Z axis
                
                print(f"🔍 DEBUG: Vertical tag normal vector: {normal_vector}, rotation_z: {rotation_z}°")
                
                # Camera offset from tag in tag coordinates
                camera_offset_x = cam_pos_in_tag[0]  # Horizontal offset along wall
                camera_distance_from_wall = cam_pos_in_tag[1] * self.distance_scale_factor  # Apply distance scaling
                camera_height = cam_pos_in_tag[2]  # Height relative to tag center
                
                print(f"🔍 DEBUG: Camera relative to vertical tag: X={camera_offset_x:.3f}m, Distance={camera_distance_from_wall:.3f}m (scaled by {self.distance_scale_factor}x), Height={camera_height:.3f}m")
                
                # Apply horizontal flip correction
                if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                    camera_offset_x = -camera_offset_x
                    print(f"🔍 DEBUG: Camera offset after flip correction: X={camera_offset_x:.3f}m")
                
                # Transform to world coordinates based on tag orientation
                if rotation_z == 0.0:  # Tag facing +Y (north)
                    camera_world_x = tag_world_pos[0] + camera_offset_x
                    camera_world_y = tag_world_pos[1] + camera_distance_from_wall
                elif rotation_z == 180.0:  # Tag facing -Y (south)
                    camera_world_x = tag_world_pos[0] - camera_offset_x
                    camera_world_y = tag_world_pos[1] - camera_distance_from_wall
                elif rotation_z == 90.0:  # Tag facing +X (east)
                    camera_world_x = tag_world_pos[0] + camera_distance_from_wall
                    camera_world_y = tag_world_pos[1] - camera_offset_x
                elif rotation_z == 270.0:  # Tag facing -X (west)
                    camera_world_x = tag_world_pos[0] - camera_distance_from_wall
                    camera_world_y = tag_world_pos[1] + camera_offset_x
                else:
                    # Generic rotation - use rotation matrix
                    angle_rad = math.radians(rotation_z)
                    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                    camera_world_x = tag_world_pos[0] + camera_offset_x * cos_a + camera_distance_from_wall * sin_a
                    camera_world_y = tag_world_pos[1] - camera_offset_x * sin_a + camera_distance_from_wall * cos_a
                
                camera_world_z = tag_world_pos[2] + camera_height
                
                # Calculate camera yaw for vertical tag
                # Camera looking direction in tag coordinates
                cam_z_in_tag = R_tag_to_cam.T @ np.array([0, 0, 1])
                
                # Apply flip correction to viewing direction
                if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                    cam_z_in_tag[0] = -cam_z_in_tag[0]
                
                # Transform viewing direction to world coordinates
                if rotation_z == 0.0:  # Tag facing +Y
                    world_view_x = cam_z_in_tag[0]
                    world_view_y = cam_z_in_tag[1]
                elif rotation_z == 180.0:  # Tag facing -Y
                    world_view_x = -cam_z_in_tag[0]
                    world_view_y = -cam_z_in_tag[1]
                elif rotation_z == 90.0:  # Tag facing +X
                    world_view_x = cam_z_in_tag[1]
                    world_view_y = -cam_z_in_tag[0]
                elif rotation_z == 270.0:  # Tag facing -X
                    world_view_x = -cam_z_in_tag[1]
                    world_view_y = cam_z_in_tag[0]
                else:
                    # Generic rotation
                    angle_rad = math.radians(rotation_z)
                    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                    world_view_x = cam_z_in_tag[0] * cos_a + cam_z_in_tag[1] * sin_a
                    world_view_y = -cam_z_in_tag[0] * sin_a + cam_z_in_tag[1] * cos_a
                
                yaw_rad = math.atan2(world_view_y, world_view_x)
                yaw_deg = math.degrees(yaw_rad)
                
            else:
                # HORIZONTAL TAG (on floor) - fix coordinate transformation
                camera_height_above_tag = cam_pos_in_tag[2] * self.distance_scale_factor  # Apply distance scaling to height
                camera_offset_x = cam_pos_in_tag[0]
                camera_offset_y = cam_pos_in_tag[1]
                
                print(f"🔍 DEBUG: Camera relative to horizontal tag: X={camera_offset_x:.3f}m, Y={camera_offset_y:.3f}m, Height={camera_height_above_tag:.3f}m (scaled by {self.distance_scale_factor}x)")
                
                # Apply horizontal flip correction
                if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                    camera_offset_x = -camera_offset_x
                    print(f"🔍 DEBUG: Camera offset after flip correction: X={camera_offset_x:.3f}m")
                
                # CRITICAL FIX: For horizontal tags, the coordinate transformation needs to account for
                # the fact that the camera is looking DOWN at the tag, not AT the tag
                # The tag's coordinate system has Z pointing up, but we need world coordinates
                
                # Transform camera position to world coordinates
                # For a horizontal tag lying flat: tag X,Y are world X,Y, tag Z is height
                camera_world_x = tag_world_pos[0] + camera_offset_x
                camera_world_y = tag_world_pos[1] + camera_offset_y  
                camera_world_z = tag_world_pos[2] + camera_height_above_tag
                
                print(f"🔍 DEBUG: World position calculation:")
                print(f"   Tag world pos: ({tag_world_pos[0]}, {tag_world_pos[1]}, {tag_world_pos[2]})")
                print(f"   Camera offset: ({camera_offset_x:.3f}, {camera_offset_y:.3f}, {camera_height_above_tag:.3f})")
                print(f"   Result: ({camera_world_x:.3f}, {camera_world_y:.3f}, {camera_world_z:.3f})")
                
                # Calculate camera yaw for horizontal tag
                # Camera looking direction in tag coordinates
                cam_z_in_tag = R_tag_to_cam.T @ np.array([0, 0, 1])
                print(f"🔍 DEBUG: Camera Z direction in tag coords: {cam_z_in_tag}")
                
                # Apply flip correction to viewing direction
                if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                    cam_z_in_tag[0] = -cam_z_in_tag[0]
                    print(f"🔍 DEBUG: Camera Z direction after flip: {cam_z_in_tag}")
                
                # For horizontal tags, calculate yaw from horizontal components
                # The camera is looking down, so we project the viewing direction onto XY plane
                yaw_rad = math.atan2(cam_z_in_tag[1], cam_z_in_tag[0])
                yaw_deg = math.degrees(yaw_rad)
                
                # Normalize yaw to 0-360 range
                if yaw_deg < 0:
                    yaw_deg += 360
                
                print(f"🔍 DEBUG: Calculated yaw: {yaw_deg:.1f}°")
            
            # Final camera position in world coordinates
            camera_world_pos = np.array([camera_world_x, camera_world_y, camera_world_z])
            print(f"🔍 DEBUG: camera_world_pos: {camera_world_pos}")
            print(f"🔍 DEBUG: yaw_deg (for {orientation} tag): {yaw_deg}")
            
            # Estimate FOV based on tag size in image
            tag_pixel_width = np.max(corners[:, 0]) - np.min(corners[:, 0])
            
            if camera_world_z > 0 and tag_pixel_width > 0:
                tag_angular_width = 2 * math.atan(tag_size / (2 * camera_world_z))
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
                'distance_to_tag': float(np.linalg.norm(tvec)),
                'reference_tag_id': tag_id,
                'reference_tag_pos': tag_world_pos.tolist(),
                'reference_tag_orientation': orientation.upper(),
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

    def estimate_2d_camera_position(self, tag_info: dict, corners: np.ndarray, frame_shape: tuple) -> Optional[Dict[str, float]]:
        """
        Simplified 2D camera position estimation for wall-mounted tags.
        
        Args:
            tag_info: Tag information
            corners: Detected tag corners in image
            frame_shape: (height, width) of the frame
            
        Returns:
            2D position dict or None
        """
        try:
            # Get tag world position (2D)
            tag_world_x = tag_info['position'][0]
            tag_world_y = tag_info['position'][1]
            tag_size = tag_info['size']
            
            print(f"🔍 2D DEBUG: Tag world position: ({tag_world_x}, {tag_world_y})")
            print(f"🔍 2D DEBUG: Tag size: {tag_size}m")
            
            # Calculate tag center and size in image
            tag_center_x = np.mean(corners[:, 0])
            tag_center_y = np.mean(corners[:, 1])
            tag_width_pixels = np.max(corners[:, 0]) - np.min(corners[:, 0])
            tag_height_pixels = np.max(corners[:, 1]) - np.min(corners[:, 1])
            
            print(f"🔍 2D DEBUG: Tag in image - center: ({tag_center_x:.1f}, {tag_center_y:.1f}), size: {tag_width_pixels:.1f}x{tag_height_pixels:.1f}px")
            
            # Estimate distance to tag based on apparent size
            # Assuming focal length from config
            focal_length = self.focal_length
            estimated_distance = (tag_size * focal_length) / tag_width_pixels
            estimated_distance *= self.distance_scale_factor  # Apply distance scaling for better visualization
            
            print(f"🔍 2D DEBUG: Estimated distance to tag: {estimated_distance:.3f}m (scaled by {self.distance_scale_factor}x)")
            
            # Calculate horizontal offset from tag center
            frame_center_x = frame_shape[1] / 2
            pixel_offset_x = tag_center_x - frame_center_x
            
            # Convert pixel offset to world offset (simplified)
            # This is an approximation - more accurate with proper camera calibration
            horizontal_fov_rad = math.radians(70)  # Assume 70° FOV
            pixels_per_radian = frame_shape[1] / horizontal_fov_rad
            angle_offset_rad = pixel_offset_x / pixels_per_radian
            
            # Calculate camera position relative to tag
            horizontal_offset = estimated_distance * math.tan(angle_offset_rad)
            
            print(f"🔍 2D DEBUG: Horizontal offset from tag: {horizontal_offset:.3f}m")
            
            # Apply horizontal flip correction
            if hasattr(self, '_is_horizontally_flipped') and self._is_horizontally_flipped:
                horizontal_offset = -horizontal_offset
                print(f"🔍 2D DEBUG: Horizontal offset after flip correction: {horizontal_offset:.3f}m")
            
            # Calculate camera world position (2D)
            # For wall-mounted tags facing north (positive Y direction)
            camera_world_x = tag_world_x + horizontal_offset
            camera_world_y = tag_world_y + estimated_distance  # Camera is in front of wall
            
            print(f"🔍 2D DEBUG: Camera 2D position: ({camera_world_x:.3f}, {camera_world_y:.3f})")
            
            # Calculate camera yaw (simplified)
            # Camera is looking toward the wall (negative Y direction)
            yaw_deg = 270.0  # Facing north wall
            if horizontal_offset != 0:
                # Adjust yaw based on horizontal offset
                yaw_adjustment = math.degrees(math.atan(horizontal_offset / estimated_distance))
                yaw_deg += yaw_adjustment
            
            # Normalize yaw
            yaw_deg = yaw_deg % 360
            
            print(f"🔍 2D DEBUG: Camera yaw: {yaw_deg:.1f}°")
            
            return {
                'x': camera_world_x,
                'y': camera_world_y,
                'z': 1.5,  # Assume camera height
                'yaw_deg': yaw_deg,
                'distance_to_tag': estimated_distance,
                'horizontal_offset': horizontal_offset
            }
            
        except Exception as e:
            print(f"❌ 2D positioning error: {e}")
            return None


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
        
        # Distance scaling for visualization (makes camera appear further from tags)
        self.distance_scale_factor = 1.5  # Multiply calculated distance by this factor
        print(f"📏 Distance scale factor: {self.distance_scale_factor}x (for better visualization)")
        
        # AprilTag pose estimation (initialize after distance_scale_factor is defined)
        self.apriltag_estimator = None
        if enable_apriltag_pose:
            try:
                self.apriltag_estimator = AprilTagPoseEstimator(self.pose['cam_id'], apriltag_config)
                # Set distance scaling factor for better visualization
                self.apriltag_estimator.distance_scale_factor = self.distance_scale_factor
                print(f"🏷️  AprilTag pose estimation enabled (config: {apriltag_config}, update interval: {apriltag_update_interval}s)")
            except Exception as e:
                print(f"❌ Failed to initialize AprilTag estimator: {e}")
                self.enable_apriltag_pose = False
        
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
        
        # Pose smoothing parameters
        self.pose_smoothing_enabled = True
        self.pose_smoothing_alpha = 0.3  # Exponential smoothing factor (0.1 = very smooth, 0.9 = very reactive)
        self.smoothed_pose = self.pose.copy()  # Current smoothed pose
        self.raw_pose_history = []  # Keep history of raw poses for debugging
        self.max_position_change = 0.5  # Maximum position change per update (meters)
        self.max_yaw_change = 30.0  # Maximum yaw change per update (degrees)
        
        print(f"🎛️  Pose smoothing enabled with alpha={self.pose_smoothing_alpha} (lower = smoother)")
        print(f"   Max position change: {self.max_position_change}m per update")
        print(f"   Max yaw change: {self.max_yaw_change}° per update")
    
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
    
    def smooth_pose_update(self, new_pose: dict) -> dict:
        """
        Apply smoothing to pose updates to reduce jitter and abrupt movements.
        
        Args:
            new_pose: Raw pose estimate from AprilTag detection
            
        Returns:
            Smoothed pose dictionary
        """
        if not self.pose_smoothing_enabled:
            return new_pose
        
        # Store raw pose for debugging
        raw_pose_entry = {
            'timestamp': time.time(),
            'x': new_pose['x'],
            'y': new_pose['y'],
            'yaw_deg': new_pose['yaw_deg'],
            'reference_tag_id': new_pose.get('reference_tag_id', 'unknown')
        }
        self.raw_pose_history.append(raw_pose_entry)
        
        # Keep only last 10 raw poses for debugging
        if len(self.raw_pose_history) > 10:
            self.raw_pose_history.pop(0)
        
        # Get current smoothed position
        current_x = self.smoothed_pose['x']
        current_y = self.smoothed_pose['y'] 
        current_yaw = self.smoothed_pose['yaw_deg']
        
        # New raw position
        new_x = new_pose['x']
        new_y = new_pose['y']
        new_yaw = new_pose['yaw_deg']
        
        # Calculate position change
        position_change = math.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
        
        # Calculate yaw change (handle wraparound)
        yaw_diff = new_yaw - current_yaw
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360
        yaw_change = abs(yaw_diff)
        
        # Apply maximum change limits to prevent sudden jumps
        if position_change > self.max_position_change:
            # Limit the position change
            scale_factor = self.max_position_change / position_change
            new_x = current_x + (new_x - current_x) * scale_factor
            new_y = current_y + (new_y - current_y) * scale_factor
            print(f"🚧 Position change limited: {position_change:.3f}m -> {self.max_position_change:.3f}m")
        
        if yaw_change > self.max_yaw_change:
            # Limit the yaw change
            scale_factor = self.max_yaw_change / yaw_change
            yaw_diff_limited = yaw_diff * scale_factor
            new_yaw = current_yaw + yaw_diff_limited
            print(f"🚧 Yaw change limited: {yaw_change:.1f}° -> {self.max_yaw_change:.1f}°")
        
        # Apply exponential smoothing
        alpha = self.pose_smoothing_alpha
        smoothed_x = alpha * new_x + (1 - alpha) * current_x
        smoothed_y = alpha * new_y + (1 - alpha) * current_y
        smoothed_yaw = alpha * new_yaw + (1 - alpha) * current_yaw
        
        # Normalize yaw to 0-360 range
        smoothed_yaw = smoothed_yaw % 360
        
        # Create smoothed pose
        smoothed_pose = new_pose.copy()
        smoothed_pose.update({
            'x': smoothed_x,
            'y': smoothed_y,
            'yaw_deg': smoothed_yaw,
            'smoothing_applied': True,
            'raw_x': new_pose['x'],
            'raw_y': new_pose['y'],
            'raw_yaw_deg': new_pose['yaw_deg'],
            'position_change': position_change,
            'yaw_change': yaw_change,
            'smoothing_alpha': alpha
        })
        
        # Update internal smoothed pose
        self.smoothed_pose.update({
            'x': smoothed_x,
            'y': smoothed_y,
            'yaw_deg': smoothed_yaw,
            'fov_deg': new_pose.get('fov_deg', self.smoothed_pose['fov_deg'])
        })
        
        # Debug logging for smoothing
        print(f"🎛️  POSE SMOOTHING for Camera {self.pose['cam_id']}:")
        print(f"   Raw pose: ({new_pose['x']:.3f}, {new_pose['y']:.3f}) @ {new_pose['yaw_deg']:.1f}°")
        print(f"   Smoothed: ({smoothed_x:.3f}, {smoothed_y:.3f}) @ {smoothed_yaw:.1f}°")
        print(f"   Changes: pos={position_change:.3f}m, yaw={yaw_change:.1f}°")
        print(f"   Alpha: {alpha}, Limits: pos={self.max_position_change}m, yaw={self.max_yaw_change}°")
        
        return smoothed_pose
    
    def get_pose_smoothing_stats(self) -> dict:
        """Get statistics about pose smoothing."""
        if not self.raw_pose_history:
            return {'message': 'No pose history available'}
        
        # Calculate position variance from recent poses
        if len(self.raw_pose_history) >= 2:
            positions = [(p['x'], p['y']) for p in self.raw_pose_history]
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
            y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
            position_variance = x_variance + y_variance
            
            yaws = [p['yaw_deg'] for p in self.raw_pose_history]
            yaw_variance = np.var(yaws) if len(yaws) > 1 else 0
            
            return {
                'raw_poses_count': len(self.raw_pose_history),
                'position_variance': position_variance,
                'yaw_variance': yaw_variance,
                'smoothing_alpha': self.pose_smoothing_alpha,
                'current_smoothed_pose': (self.smoothed_pose['x'], self.smoothed_pose['y'], self.smoothed_pose['yaw_deg']),
                'latest_raw_pose': (self.raw_pose_history[-1]['x'], self.raw_pose_history[-1]['y'], self.raw_pose_history[-1]['yaw_deg'])
            }
        
        return {'message': 'Insufficient pose history for statistics'}

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
            
            # Apply pose smoothing before updating current pose
            smoothed_apriltag_pose = self.smooth_pose_update(apriltag_pose)
            
            # Update current pose with smoothed AprilTag-based pose
            old_pose = self.pose.copy()
            
            self.pose.update({
                'x': smoothed_apriltag_pose['x'],
                'y': smoothed_apriltag_pose['y'], 
                'yaw_deg': smoothed_apriltag_pose['yaw_deg'],
                'fov_deg': smoothed_apriltag_pose['fov_deg']
            })
            
            self.last_apriltag_update = current_time
            self.apriltag_update_count += 1
            
            # Calculate position change (using smoothed values)
            position_change = math.sqrt((smoothed_apriltag_pose['x'] - old_pose['x'])**2 + (smoothed_apriltag_pose['y'] - old_pose['y'])**2)
            yaw_change = abs(smoothed_apriltag_pose['yaw_deg'] - old_pose['yaw_deg'])
            if yaw_change > 180:
                yaw_change = 360 - yaw_change  # Handle wraparound
            
            # Get detection statistics summary
            detection_summary = self.apriltag_estimator.get_detection_summary()
            
            print(f"\n🏷️  APRILTAG POSE UPDATE #{self.apriltag_update_count} for Camera {self.pose['cam_id']}:")
            print(f"   📍 CAMERA POSITION:")
            print(f"      Old: ({old_pose['x']:.3f}, {old_pose['y']:.3f}) m")
            print(f"      Raw: ({apriltag_pose['x']:.3f}, {apriltag_pose['y']:.3f}) m")
            print(f"      Smoothed: ({self.pose['x']:.3f}, {self.pose['y']:.3f}) m")
            print(f"      Change: {position_change:.3f}m")
            print(f"   🧭 CAMERA ORIENTATION:")
            print(f"      Old: {old_pose['yaw_deg']:.1f}°")
            print(f"      Raw: {apriltag_pose['yaw_deg']:.1f}°")
            print(f"      Smoothed: {self.pose['yaw_deg']:.1f}°")
            print(f"      Change: {yaw_change:.1f}°")
            print(f"   🎛️  SMOOTHING STATUS:")
            print(f"      Enabled: {'YES' if self.pose_smoothing_enabled else 'NO'}")
            print(f"      Alpha: {self.pose_smoothing_alpha} (lower = smoother)")
            print(f"      Raw position change: {smoothed_apriltag_pose.get('position_change', 'N/A'):.3f}m")
            print(f"      Raw yaw change: {smoothed_apriltag_pose.get('yaw_change', 'N/A'):.1f}°")
            print(f"   🔄 COORDINATE SYSTEM:")
            print(f"      Horizontal flip: {'ENABLED' if self.flip_horizontal else 'DISABLED'}")
            print(f"      Flip correction applied: {'YES' if hasattr(self.apriltag_estimator, '_is_horizontally_flipped') and self.apriltag_estimator._is_horizontally_flipped else 'NO'}")
            tag_orientation = apriltag_pose.get('reference_tag_orientation', 'UNKNOWN')
            print(f"      Tag orientation: {tag_orientation}")
            print(f"      Camera height above floor: {apriltag_pose.get('z', 0):.3f}m")
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
            "pose_broadcast": True,  # Flag to indicate this is an immediate pose update
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
                    
                    # Show pose smoothing status
                    if self.pose_smoothing_enabled:
                        smoothing_stats = self.get_pose_smoothing_stats()
                        if 'position_variance' in smoothing_stats:
                            smoothing_text = f"Smoothing: α={self.pose_smoothing_alpha} | Pos Var: {smoothing_stats['position_variance']:.4f} | Yaw Var: {smoothing_stats['yaw_variance']:.2f}"
                            cv2.putText(display_frame, smoothing_text,
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                            y_offset += 20
                    else:
                        smoothing_text = "Smoothing: DISABLED (immediate updates)"
                        cv2.putText(display_frame, smoothing_text,
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        y_offset += 20
                    
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
    
    # Pose smoothing arguments
    parser.add_argument('--disable-pose-smoothing', action='store_true',
                       help='Disable pose smoothing (poses will update immediately)')
    parser.add_argument('--smoothing-alpha', type=float, default=0.3,
                       help='Pose smoothing factor (0.1=very smooth, 0.9=very reactive, default: 0.3)')
    parser.add_argument('--max-position-change', type=float, default=0.5,
                       help='Maximum position change per update in meters (default: 0.5m)')
    parser.add_argument('--max-yaw-change', type=float, default=30.0,
                       help='Maximum yaw change per update in degrees (default: 30°)')
    
    # Distance scaling for visualization
    parser.add_argument('--distance-scale', type=float, default=1.5,
                       help='Distance scaling factor for visualization (default: 1.5x, makes camera appear further from tags)')
    
    # AprilTag detection tuning
    parser.add_argument('--apriltag-confidence', type=float, default=None,
                       help='Override AprilTag detection confidence threshold (default: use config file value)')
    parser.add_argument('--apriltag-debug', action='store_true',
                       help='Enable extra verbose AprilTag debugging output')
    
    args = parser.parse_args()
    
    # Flip is enabled by default, disabled only if --no-flip is specified
    flip_horizontal = not args.no_flip
    
    if args.enable_apriltag_pose:
        print(f"🏷️  AprilTag pose estimation enabled:")
        print(f"   Config file: {args.apriltag_config}")
        print(f"   Update interval: {args.apriltag_update_interval}s")
        print(f"   💡 Place AprilTags at the positions specified in {args.apriltag_config}")
        print(f"   💡 Camera pose will automatically update every {args.apriltag_update_interval}s when tags are detected")
        
        # Show smoothing configuration
        if not args.disable_pose_smoothing:
            print(f"🎛️  Pose smoothing enabled:")
            print(f"   Smoothing alpha: {args.smoothing_alpha} (lower = smoother)")
            print(f"   Max position change: {args.max_position_change}m per update")
            print(f"   Max yaw change: {args.max_yaw_change}° per update")
            print(f"   💡 Adjust --smoothing-alpha for more/less responsiveness")
        else:
            print(f"🎛️  Pose smoothing disabled - poses will update immediately")
    
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
        
        # Configure pose smoothing if AprilTag pose estimation is enabled
        if args.enable_apriltag_pose:
            client.pose_smoothing_enabled = not args.disable_pose_smoothing
            client.pose_smoothing_alpha = args.smoothing_alpha
            client.max_position_change = args.max_position_change
            client.max_yaw_change = args.max_yaw_change
            
            # Configure distance scaling
            client.distance_scale_factor = args.distance_scale
            if client.apriltag_estimator:
                client.apriltag_estimator.distance_scale_factor = args.distance_scale
                
                # Configure AprilTag detection parameters
                if args.apriltag_confidence is not None:
                    old_confidence = client.apriltag_estimator.min_detection_confidence
                    client.apriltag_estimator.min_detection_confidence = args.apriltag_confidence
                    print(f"🔧 AprilTag confidence threshold overridden: {old_confidence} -> {args.apriltag_confidence}")
                
                # Enable verbose debugging if requested
                client.apriltag_estimator.verbose_debug = args.apriltag_debug
                if args.apriltag_debug:
                    print(f"🔧 Extra verbose AprilTag debugging enabled")
            
            print(f"🔧 Pose smoothing configured:")
            print(f"   Enabled: {client.pose_smoothing_enabled}")
            if client.pose_smoothing_enabled:
                print(f"   Alpha: {client.pose_smoothing_alpha}")
                print(f"   Max position change: {client.max_position_change}m")
                print(f"   Max yaw change: {client.max_yaw_change}°")
            print(f"📏 Distance scaling: {client.distance_scale_factor}x (camera will appear further from tags)")
        
        await client.run_detection_loop()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 