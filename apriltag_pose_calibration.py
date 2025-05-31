"""
AprilTag-based camera pose calibration.
Uses AprilTag detection to estimate camera pose relative to known tag positions.
"""

import cv2
import json
import time
import argparse
import numpy as np
import math
from typing import Optional, Dict, Any, List, Tuple
import os


class AprilTagPoseEstimator:
    """AprilTag-based pose estimation for camera calibration."""
    
    def __init__(self, config_file: str = "apriltag_config.json"):
        """
        Initialize AprilTag pose estimator.
        
        Args:
            config_file: Path to AprilTag configuration JSON file
        """
        # Load configuration
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"AprilTag config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        print(f"📋 Loaded AprilTag configuration from {config_file}")
        print(f"   Tag family: {self.config['tag_family']}")
        print(f"   Number of tags: {len(self.config['apriltags'])}")
        
        # Create tag position lookup
        self.tag_positions = {}
        for tag in self.config['apriltags']:
            tag_id = tag['id']
            self.tag_positions[tag_id] = {
                'position': np.array([tag['x'], tag['y'], tag['z']], dtype=np.float32),
                'size': tag.get('size', self.config['default_size']),
                'description': tag.get('description', f"Tag {tag_id}")
            }
            print(f"   Tag {tag_id}: {tag['description']} - size: {tag.get('size', self.config['default_size'])}m")
        
        # Camera intrinsic parameters (will be updated when resolution is known)
        self.focal_length = self.config['camera_intrinsics']['focal_length']
        self.dist_coeffs = np.array(self.config['camera_intrinsics']['distortion_coeffs'], dtype=np.float32)
        self.camera_matrix = None
        
        # Initialize AprilTag detector using robotpy_apriltag
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
            
            print(f"✅ RobotPy AprilTag detector initialized with family {self.config['tag_family']}")
        except ImportError:
            raise ImportError("robotpy-apriltag library not found. Install with: pip install robotpy-apriltag")
        
        # Pose estimation parameters
        self.max_reproj_error = self.config['pose_estimation']['max_reproj_error']
        self.min_detection_confidence = self.config['pose_estimation']['min_detection_confidence']
        
        # Detection statistics
        self.detection_stats = {
            'total_frames': 0,
            'frames_with_tags': 0,
            'total_tags_detected': 0,
            'successful_poses': 0,
            'failed_poses': 0,
            'tag_detection_counts': {},
            'last_detection_time': 0,
            'detection_rate_history': []
        }
        
        print(f"🔧 Detection confidence threshold: {self.min_detection_confidence}")
        print(f"🔧 Max reprojection error: {self.max_reproj_error}")
    
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
            List of detection dictionaries
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
        
        # Log detection summary periodically
        if self.detection_stats['total_frames'] % 30 == 0:  # Every 30 frames
            self._log_detection_status(detected_tags, rejected_tags, detection_time)
        
        return detected_tags
    
    def _log_detection_status(self, detected_tags: List[Dict], rejected_tags: List[Dict], detection_time: float):
        """Log detailed detection status."""
        stats = self.detection_stats
        detection_rate = (stats['frames_with_tags'] / stats['total_frames']) * 100 if stats['total_frames'] > 0 else 0
        
        print(f"\n📊 APRILTAG DETECTION STATUS:")
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
        print(f"   Tag detection history:")
        for tag_id, counts in stats['tag_detection_counts'].items():
            success_rate = (counts['accepted'] / counts['seen']) * 100 if counts['seen'] > 0 else 0
            print(f"      Tag {tag_id}: {counts['accepted']}/{counts['seen']} ({success_rate:.1f}% success)")
        
        print("=" * 50)
    
    def estimate_pose_from_tags(self, detected_tags: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Estimate camera pose from detected AprilTags.
        
        Args:
            detected_tags: List of detected tag dictionaries
            
        Returns:
            Pose dictionary or None if estimation fails
        """
        if not detected_tags or self.camera_matrix is None:
            self.detection_stats['failed_poses'] += 1
            return None
        
        # Use the first detected tag for pose estimation
        # In the future, we could combine multiple tags for better accuracy
        tag = detected_tags[0]
        tag_id = tag['id']
        corners = tag['corners']
        tag_info = tag['tag_info']
        
        # Get tag size and world position
        tag_size = tag_info['size']
        tag_world_pos = tag_info['position']
        
        print(f"🔍 Estimating pose from Tag {tag_id}:")
        print(f"   Tag world position: ({tag_world_pos[0]}, {tag_world_pos[1]}, {tag_world_pos[2]})")
        print(f"   Tag size: {tag_size}m")
        print(f"   Detection confidence: {tag['confidence']:.3f}")
        print(f"   Tag area in image: {tag['area']:.0f}px²")
        
        # Get 3D object points in tag coordinate system
        tag_object_points = self.get_tag_object_points(tag_size)
        
        try:
            # Solve PnP to get tag pose relative to camera
            start_time = time.time()
            success, rvec, tvec = cv2.solvePnP(
                tag_object_points,
                corners,
                self.camera_matrix,
                self.dist_coeffs
            )
            solve_time = time.time() - start_time
            
            if not success:
                print(f"   ❌ solvePnP failed")
                self.detection_stats['failed_poses'] += 1
                return None
            
            print(f"   ✅ solvePnP successful ({solve_time*1000:.1f}ms)")
            
            # Convert rotation vector to rotation matrix
            R_tag_to_cam, _ = cv2.Rodrigues(rvec)
            t_tag_to_cam = tvec.flatten()
            
            # Transform from tag coordinate system to world coordinate system
            # Camera position in tag coordinates
            cam_pos_in_tag = -R_tag_to_cam.T @ t_tag_to_cam
            
            # Camera position in world coordinates
            # For now, assume tag is axis-aligned (no rotation)
            # This could be extended to handle rotated tags
            camera_world_pos = tag_world_pos + cam_pos_in_tag
            
            # Calculate camera orientation
            # Camera Z-axis in tag coordinates (pointing toward tag)
            cam_z_in_tag = R_tag_to_cam.T @ np.array([0, 0, 1])
            
            # Project to XY plane and calculate yaw
            yaw_rad = math.atan2(-cam_z_in_tag[1], -cam_z_in_tag[0])
            yaw_deg = math.degrees(yaw_rad)
            
            # Calculate distance to tag
            distance_to_tag = float(np.linalg.norm(t_tag_to_cam))
            
            # Estimate FOV based on tag size in image
            tag_pixel_width = np.max(corners[:, 0]) - np.min(corners[:, 0])
            
            if distance_to_tag > 0 and tag_pixel_width > 0:
                tag_angular_width = 2 * math.atan(tag_size / (2 * distance_to_tag))
                estimated_fov = math.degrees(tag_angular_width) * (self.camera_matrix[0, 2] * 2 / tag_pixel_width)
                estimated_fov = max(30.0, min(120.0, estimated_fov))
            else:
                estimated_fov = 70.0
            
            # Calculate reprojection error for quality assessment
            projected_points, _ = cv2.projectPoints(
                tag_object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = cv2.norm(corners.reshape(-1, 1, 2), projected_points, cv2.NORM_L2) / len(corners)
            
            pose = {
                'x': float(camera_world_pos[0]),
                'y': float(camera_world_pos[1]),
                'z': float(camera_world_pos[2]),
                'yaw_deg': float(yaw_deg),
                'fov_deg': float(estimated_fov),
                'distance_to_tag': distance_to_tag,
                'reference_tag_id': tag_id,
                'reference_tag_pos': tag_world_pos.tolist(),
                'confidence': tag['confidence'],
                'reprojection_error': float(reprojection_error),
                'calibration_method': 'apriltag_realtime',
                'timestamp': time.time()
            }
            
            # Log pose estimation results
            print(f"   📍 Estimated camera position: ({camera_world_pos[0]:.3f}, {camera_world_pos[1]:.3f}, {camera_world_pos[2]:.3f})")
            print(f"   🧭 Estimated camera yaw: {yaw_deg:.1f}°")
            print(f"   📏 Distance to tag: {distance_to_tag:.3f}m")
            print(f"   📐 Estimated FOV: {estimated_fov:.1f}°")
            print(f"   🎯 Reprojection error: {reprojection_error:.2f}px")
            
            if reprojection_error > self.max_reproj_error:
                print(f"   ⚠️  High reprojection error (>{self.max_reproj_error}px) - pose may be inaccurate")
            
            self.detection_stats['successful_poses'] += 1
            return pose
            
        except Exception as e:
            print(f"   ❌ Pose estimation error: {e}")
            self.detection_stats['failed_poses'] += 1
            return None
    
    def estimate_pose_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Main interface: detect tags and estimate pose from frame.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Pose dictionary or None if estimation fails
        """
        detected_tags = self.detect_apriltags(frame)
        
        if not detected_tags:
            return None
        
        return self.estimate_pose_from_tags(detected_tags)
    
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
    
    def draw_detections(self, frame: np.ndarray, detected_tags: List[Dict]) -> np.ndarray:
        """
        Draw detected AprilTags on frame for visualization.
        
        Args:
            frame: Input frame
            detected_tags: List of detected tags
            
        Returns:
            Frame with drawn detections
        """
        display_frame = frame.copy()
        
        for tag in detected_tags:
            tag_id = tag['id']
            corners = tag['corners']
            center = tag['center']
            confidence = tag['confidence']
            area = tag.get('area', 0)
            
            # Draw tag corners
            corners_int = corners.astype(int)
            cv2.polylines(display_frame, [corners_int], True, (0, 255, 0), 2)
            
            # Draw tag ID and confidence
            cv2.putText(display_frame, f"ID:{tag_id}", 
                       (int(center[0]) - 30, int(center[1]) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"C:{confidence:.2f}", 
                       (int(center[0]) - 30, int(center[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, f"A:{area:.0f}", 
                       (int(center[0]) - 30, int(center[1]) + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw center point
            cv2.circle(display_frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        
        return display_frame


def run_calibration_session(camera_id: str, config_file: str, output_file: str, num_samples: int = 10):
    """
    Run interactive calibration session using AprilTag detection.
    
    Args:
        camera_id: Camera identifier for output pose file
        config_file: AprilTag configuration file
        output_file: Output pose JSON file
        num_samples: Number of pose samples to average
    """
    print(f"🎯 Starting AprilTag pose calibration for camera {camera_id}")
    print(f"📄 Using config: {config_file}")
    print(f"💾 Output file: {output_file}")
    print(f"📊 Will average {num_samples} pose measurements")
    
    # Initialize pose estimator
    estimator = AprilTagPoseEstimator(config_file)
    
    # Find and open camera
    cap = None
    for cam_idx in range(5):
        test_cap = cv2.VideoCapture(cam_idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                print(f"📹 Using camera index {cam_idx}")
                cap = test_cap
                break
        test_cap.release()
    
    if cap is None:
        raise RuntimeError("No working camera found")
    
    # Get camera resolution and update estimator
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from camera")
    
    height, width = frame.shape[:2]
    estimator.set_camera_resolution(width, height)
    print(f"📐 Camera resolution: {width}x{height}")
    
    # Calibration loop
    poses = []
    frame_count = 0
    start_time = time.time()
    last_status_log = 0
    
    print(f"\n🔄 Starting calibration session...")
    print(f"📺 Position camera to see AprilTags and press SPACE to capture pose")
    print(f"📺 Press 'q' to quit early or 'r' to reset samples")
    print(f"📺 Status logs will appear every 30 frames")
    print("=" * 60)
    
    while len(poses) < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Detect tags and estimate pose
        detected_tags = estimator.detect_apriltags(frame)
        current_pose = estimator.estimate_pose_from_tags(detected_tags) if detected_tags else None
        
        # Draw detections
        display_frame = estimator.draw_detections(frame, detected_tags)
        
        # Show status
        status_text = f"Samples: {len(poses)}/{num_samples} | Frame: {frame_count}"
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show detection stats
        current_time = time.time()
        elapsed = current_time - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        stats_text = f"FPS: {fps:.1f} | Tags: {len(detected_tags)}"
        cv2.putText(display_frame, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if detected_tags:
            pose_text = f"Tags detected: {len(detected_tags)}"
            cv2.putText(display_frame, pose_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show tag IDs
            tag_ids = [str(tag['id']) for tag in detected_tags]
            tag_text = f"IDs: {', '.join(tag_ids)}"
            cv2.putText(display_frame, tag_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if current_pose:
                pos_text = f"Pos: ({current_pose['x']:.2f}, {current_pose['y']:.2f})"
                cv2.putText(display_frame, pos_text, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                orient_text = f"Yaw: {current_pose['yaw_deg']:.1f}°"
                cv2.putText(display_frame, orient_text, (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                error_text = f"Error: {current_pose.get('reprojection_error', 0):.2f}px"
                cv2.putText(display_frame, error_text, (10, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                ready_text = "READY - Press SPACE to capture"
                cv2.putText(display_frame, ready_text, (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                error_text = "Pose estimation failed"
                cv2.putText(display_frame, error_text, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            error_text = "No AprilTags detected"
            cv2.putText(display_frame, error_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow(f"AprilTag Calibration - Camera {camera_id}", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Calibration cancelled by user")
            break
        elif key == ord('r'):
            poses.clear()
            print(f"🔄 Reset samples - starting over")
        elif key == ord(' ') and current_pose:
            poses.append(current_pose)
            print(f"✅ Captured pose {len(poses)}/{num_samples}: "
                  f"pos=({current_pose['x']:.3f}, {current_pose['y']:.3f}), "
                  f"yaw={current_pose['yaw_deg']:.1f}°, "
                  f"tag={current_pose['reference_tag_id']}, "
                  f"error={current_pose.get('reprojection_error', 0):.2f}px")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final detection statistics
    print(f"\n📊 FINAL CALIBRATION STATISTICS:")
    summary = estimator.get_detection_summary()
    print(f"   Total frames processed: {summary['total_frames']}")
    print(f"   Detection rate: {summary['detection_rate_percent']:.1f}%")
    print(f"   Total tags detected: {summary['total_tags_detected']}")
    print(f"   Successful poses: {summary['successful_poses']}")
    print(f"   Failed poses: {summary['failed_poses']}")
    print(f"   Pose success rate: {summary['pose_success_rate_percent']:.1f}%")
    
    print(f"\n   Per-tag detection statistics:")
    for tag_id, counts in summary['tag_detection_counts'].items():
        success_rate = (counts['accepted'] / counts['seen']) * 100 if counts['seen'] > 0 else 0
        print(f"      Tag {tag_id}: {counts['accepted']}/{counts['seen']} ({success_rate:.1f}% success)")
    
    if not poses:
        print("❌ No poses captured - calibration failed")
        return False
    
    # Calculate average pose
    print(f"\n📊 Processing {len(poses)} pose samples...")
    
    avg_x = np.mean([p['x'] for p in poses])
    avg_y = np.mean([p['y'] for p in poses])
    avg_z = np.mean([p['z'] for p in poses])
    
    # Average angles carefully (handle wraparound)
    angles = [math.radians(p['yaw_deg']) for p in poses]
    avg_angle = math.atan2(np.mean([math.sin(a) for a in angles]),
                          np.mean([math.cos(a) for a in angles]))
    avg_yaw_deg = math.degrees(avg_angle)
    
    avg_fov = np.mean([p['fov_deg'] for p in poses])
    
    # Calculate standard deviations
    std_x = np.std([p['x'] for p in poses])
    std_y = np.std([p['y'] for p in poses])
    std_yaw = np.std([p['yaw_deg'] for p in poses])
    
    # Create final pose
    final_pose = {
        'cam_id': camera_id,
        'x': float(avg_x),
        'y': float(avg_y),
        'z': float(avg_z),
        'yaw_deg': float(avg_yaw_deg),
        'fov_deg': float(avg_fov),
        'img_w': width,
        'img_h': height,
        'calibration_method': 'apriltag_averaged',
        'num_samples': len(poses),
        'std_x': float(std_x),
        'std_y': float(std_y),
        'std_yaw_deg': float(std_yaw),
        'timestamp': time.time()
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(final_pose, f, indent=2)
    
    print(f"✅ Calibration complete!")
    print(f"📍 Final pose: ({avg_x:.3f}, {avg_y:.3f}) m, yaw: {avg_yaw_deg:.1f}°")
    print(f"📏 Accuracy: pos ±{max(std_x, std_y):.3f}m, yaw ±{std_yaw:.1f}°")
    print(f"💾 Saved to: {output_file}")
    
    return True


def main():
    """Main function for AprilTag pose calibration."""
    parser = argparse.ArgumentParser(description='AprilTag-based camera pose calibration')
    parser.add_argument('--camera-id', required=True, 
                       help='Camera identifier (e.g., A, B, C)')
    parser.add_argument('--config', default='apriltag_config.json',
                       help='AprilTag configuration file')
    parser.add_argument('--output', 
                       help='Output pose file (default: pose_{camera_id}.json)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of pose samples to average')
    
    args = parser.parse_args()
    
    # Generate output filename if not specified
    if args.output is None:
        args.output = f"pose_{args.camera_id}.json"
    
    try:
        success = run_calibration_session(
            args.camera_id, 
            args.config, 
            args.output, 
            args.samples
        )
        if success:
            print(f"\n🎉 Camera {args.camera_id} calibration successful!")
            print(f"🚀 You can now use: python cam_client.py --pose {args.output} --server localhost:9000 --target bottle")
        else:
            print(f"\n❌ Camera {args.camera_id} calibration failed!")
    except Exception as e:
        print(f"❌ Calibration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 