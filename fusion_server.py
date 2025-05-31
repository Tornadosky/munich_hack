"""
Fusion server for multi-camera object triangulation.
Receives detection packets via UDP and triangulates object positions using camera poses.
"""

import json
import time
import argparse
import asyncio
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glob import glob
from utils import udp_recv_async, ray_from_pixel, triangulate_two_rays


class FusionServer:
    """Multi-camera fusion server with real-time triangulation and plotting."""
    
    def __init__(self, listen_addr: str, pose_dir: str = '.'):
        """
        Initialize fusion server.
        
        Args:
            listen_addr: UDP listen address as "host:port"
            pose_dir: Directory to search for pose_*.json files (fallback only)
        """
        # Parse listen address
        host, port = listen_addr.split(':')
        self.listen_addr = (host, int(port))
        
        # Camera poses will be learned dynamically from client packets
        self.poses = {}
        
        # Track when each camera was last seen
        self.camera_last_seen = {}
        self.camera_timeout = 1.0  # Remove cameras after 1 second of no data
        
        # Try to load any existing pose files as fallback
        pose_files = glob(f"{pose_dir}/pose_*.json")
        for pose_file in pose_files:
            with open(pose_file, 'r') as f:
                pose = json.load(f)
                self.poses[pose['cam_id']] = pose
                print(f"Loaded fallback pose for camera {pose['cam_id']} (will appear when active)")
        
        if pose_files:
            print(f"Loaded {len(pose_files)} fallback poses - cameras will appear when clients connect")
        
        # Storage for latest detections per camera
        self.latest_detections = {}
        self.time_window = 2  # seconds
        
        # Storage for detection rays (for visualization)
        self.detection_rays = {}  # cam_id -> (ray_start, ray_end, timestamp)
        self.ray_timeout = 0.5  # Show rays for 2 seconds
        
        # Triangulated positions for plotting (thread-safe)
        self.positions = []
        self.timestamps = []
        self._positions_lock = threading.Lock()
        
        # Server running flag
        self.running = False
        
        # Flag to track if plot needs to be redrawn due to new cameras
        self.plot_needs_update = False
        
        # Setup matplotlib for live plotting
        self.setup_plot()
    
    def setup_plot(self):
        """Initialize matplotlib figure for real-time position plotting."""
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Object Position Triangulation')
        self.ax.grid(True)
        
        # Don't plot any cameras initially - they will appear when clients connect
        # and start sending data
        
        # Initialize empty scatter plot for object positions
        self.scatter = self.ax.scatter([], [], c='blue', s=100, alpha=1.0, zorder=5, edgecolors='black', linewidth=1)
        
        # Set initial axis limits
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.legend()
        
        # Try to bring window to front (backend-agnostic)
        try:
            # For different matplotlib backends
            manager = self.fig.canvas.manager
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'wm_attributes'):
                    # Tkinter backend
                    manager.window.wm_attributes('-topmost', True)
                    manager.window.after_idle(manager.window.attributes, '-topmost', False)
                elif hasattr(manager.window, 'activateWindow'):
                    # Qt backend
                    manager.window.activateWindow()
                    manager.window.raise_()
                elif hasattr(manager.window, 'present'):
                    # GTK backend
                    manager.window.present()
        except Exception as e:
            print(f"Could not bring window to front: {e}")
        
        self.fig.canvas.draw()
        plt.show(block=False)
    
    def draw_camera_fov(self, pose: dict, cam_id: str):
        """
        Draw camera field of view as a triangular sector.
        
        Args:
            pose: Camera pose dictionary with x, y, yaw_deg, fov_deg
            cam_id: Camera identifier for color coding
        """
        import math
        
        # Camera parameters
        cam_x, cam_y = pose['x'], pose['y']
        yaw_deg = pose['yaw_deg']
        fov_deg = pose['fov_deg']
        
        # FOV range (viewing distance)
        fov_range = 8.0  # meters - how far to draw the FOV triangle
        
        # Calculate FOV boundaries
        half_fov = fov_deg / 2.0
        left_angle = yaw_deg - half_fov
        right_angle = yaw_deg + half_fov
        
        # Convert to radians
        left_rad = math.radians(left_angle)
        right_rad = math.radians(right_angle)
        
        # Calculate FOV triangle vertices
        # Start at camera position
        vertices_x = [cam_x]
        vertices_y = [cam_y]
        
        # Left edge of FOV
        left_x = cam_x + fov_range * math.cos(left_rad)
        left_y = cam_y + fov_range * math.sin(left_rad)
        vertices_x.append(left_x)
        vertices_y.append(left_y)
        
        # Right edge of FOV  
        right_x = cam_x + fov_range * math.cos(right_rad)
        right_y = cam_y + fov_range * math.sin(right_rad)
        vertices_x.append(right_x)
        vertices_y.append(right_y)
        
        # Close the triangle
        vertices_x.append(cam_x)
        vertices_y.append(cam_y)
        
        # Choose color based on camera ID
        colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange'}
        color = colors.get(cam_id, 'gray')
        
        # Draw FOV triangle with transparency
        self.ax.fill(vertices_x, vertices_y, color=color, alpha=0.2, zorder=1, 
                    label=f'FOV {cam_id}')
        
        # Draw FOV boundary lines
        self.ax.plot(vertices_x, vertices_y, color=color, alpha=0.5, linewidth=1, zorder=2)
        
        # Draw center line (camera direction)
        center_x = cam_x + fov_range * math.cos(math.radians(yaw_deg))
        center_y = cam_y + fov_range * math.sin(math.radians(yaw_deg))
        self.ax.plot([cam_x, center_x], [cam_y, center_y], color=color, 
                    linewidth=2, alpha=0.7, zorder=3)
    
    def process_detection(self, packet: dict):
        """
        Process incoming detection packet and attempt triangulation.
        
        Args:
            packet: Detection packet with cam_id, cx, cy, timestamp, and optionally pose info
        """
        # Enhanced packet validation and debugging
        try:
            cam_id = packet['cam_id']
            timestamp = packet['timestamp']
            current_time = time.time()
            
            # Debug: Print detailed packet info
            print(f"\n🔍 Processing packet from camera {cam_id}:")
            print(f"   Timestamp: {timestamp:.3f} (age: {(current_time - timestamp)*1000:.1f}ms)")
            
            # Check if this is a test packet
            if packet.get('test', False):
                print(f"   📡 Test packet received from {cam_id} - connection OK")
                return
            
            # Check if this is a pose broadcast packet (no detection data)
            if packet.get('pose_broadcast', False):
                print(f"   📡 Pose broadcast received from {cam_id}")
                
                # Update camera pose if provided
                if 'pose' in packet:
                    pose_info = packet['pose']
                    print(f"   📐 Pose info: pos=({pose_info.get('x', '?')}, {pose_info.get('y', '?')}), yaw={pose_info.get('yaw_deg', '?')}°")
                    
                    if cam_id not in self.poses:
                        # New camera discovered via pose broadcast
                        self.poses[cam_id] = pose_info
                        print(f"   ✅ NEW CAMERA DISCOVERED (broadcast): {cam_id} at ({pose_info['x']}, {pose_info['y']}) facing {pose_info['yaw_deg']}°")
                        self.plot_needs_update = True
                    else:
                        # Update existing camera pose if it changed
                        if self.poses[cam_id] != pose_info:
                            print(f"   📝 Updated pose for camera {cam_id} (broadcast)")
                            self.poses[cam_id] = pose_info
                            self.plot_needs_update = True
                        else:
                            print(f"   ✅ Pose confirmed for camera {cam_id} (broadcast)")
                
                # If camera just became active, update plot
                if not was_active:
                    print(f"   🟢 Camera {cam_id} is now ACTIVE (via broadcast)")
                    self.plot_needs_update = True
                
                # Don't proceed with triangulation for broadcast packets
                return
            
            # Validate required fields for detection packets
            required_fields = ['cam_id', 'cx', 'cy', 'timestamp']
            missing_fields = [field for field in required_fields if field not in packet]
            if missing_fields:
                print(f"   ❌ Missing required fields: {missing_fields}")
                return
            
            cx, cy = packet['cx'], packet['cy']
            confidence = packet.get('confidence', 'unknown')
            print(f"   📍 Detection: cx={cx:.1f}, cy={cy:.1f}, confidence={confidence}")
            
        except KeyError as e:
            print(f"❌ Invalid packet structure - missing key: {e}")
            print(f"   Packet keys: {list(packet.keys())}")
            return
        except Exception as e:
            print(f"❌ Error parsing packet: {e}")
            return
        
        # Update camera last-seen timestamp
        was_active = cam_id in self.camera_last_seen and (current_time - self.camera_last_seen[cam_id]) <= self.camera_timeout
        self.camera_last_seen[cam_id] = current_time
        
        # Check if packet contains camera pose information
        if 'pose' in packet:
            pose_info = packet['pose']
            print(f"   📐 Pose info: pos=({pose_info.get('x', '?')}, {pose_info.get('y', '?')}), yaw={pose_info.get('yaw_deg', '?')}°")
            
            if cam_id not in self.poses:
                # New camera discovered
                self.poses[cam_id] = pose_info
                print(f"   ✅ NEW CAMERA DISCOVERED: {cam_id} at ({pose_info['x']}, {pose_info['y']}) facing {pose_info['yaw_deg']}°")
                self.plot_needs_update = True
            else:
                # Update existing camera pose if it changed
                if self.poses[cam_id] != pose_info:
                    print(f"   📝 Updated pose for camera {cam_id}")
                    self.poses[cam_id] = pose_info
                    self.plot_needs_update = True
                else:
                    print(f"   ✅ Pose confirmed for camera {cam_id}")
        else:
            print(f"   ⚠️  No pose info in packet")
            if cam_id not in self.poses:
                print(f"   ❌ No pose available for camera {cam_id} - cannot triangulate")
                return
        
        # If camera just became active, update plot
        if not was_active:
            print(f"   🟢 Camera {cam_id} is now ACTIVE")
            self.plot_needs_update = True
        
        # Store latest detection for this camera
        self.latest_detections[cam_id] = packet
        print(f"   💾 Stored detection for camera {cam_id}")
        
        # Calculate and store detection ray for visualization
        self.calculate_detection_ray(cam_id, cx, cy, current_time)
        
        # Get list of active cameras (have sent data recently)
        active_cameras = self.get_active_cameras()
        print(f"   📊 Active cameras: {list(active_cameras)} (total: {len(active_cameras)})")
        
        # Check if we have enough active cameras for triangulation
        if len(active_cameras) < 2:
            print(f"   ⏳ Need at least 2 active cameras for triangulation (have {len(active_cameras)})")
            return
        
        # Try triangulation with other active cameras
        triangulation_attempts = 0
        successful_triangulations = 0
        
        for other_cam_id, other_detection in self.latest_detections.items():
            if other_cam_id == cam_id:
                continue
            
            triangulation_attempts += 1
            print(f"\n   🔄 Triangulation attempt {triangulation_attempts}: {cam_id} + {other_cam_id}")
            
            # Check if other camera is active
            if other_cam_id not in active_cameras:
                print(f"      ❌ Camera {other_cam_id} not active")
                continue
            
            # Check if we have pose info for both cameras
            if cam_id not in self.poses or other_cam_id not in self.poses:
                print(f"      ❌ Missing pose info: {cam_id} in poses: {cam_id in self.poses}, {other_cam_id} in poses: {other_cam_id in self.poses}")
                continue
            
            # Check if detections are within time window
            time_diff = abs(timestamp - other_detection['timestamp'])
            print(f"      ⏱️  Time difference: {time_diff:.3f}s (limit: {self.time_window}s)")
            if time_diff > self.time_window:
                print(f"      ❌ Time difference too large")
                continue
            
            # Perform triangulation
            try:
                print(f"      🎯 Attempting triangulation...")
                print(f"         Cam {cam_id}: cx={cx:.1f}, pose=({self.poses[cam_id]['x']}, {self.poses[cam_id]['y']})")
                print(f"         Cam {other_cam_id}: cx={other_detection['cx']:.1f}, pose=({self.poses[other_cam_id]['x']}, {self.poses[other_cam_id]['y']})")
                
                position = self.triangulate_pair(packet, other_detection)
                age_ms = (current_time - min(timestamp, other_detection['timestamp'])) * 1000
                
                # Store result
                with self._positions_lock:
                    self.positions.append(position)
                    self.timestamps.append(current_time)
                    # Keep only recent positions for plotting
                    self.cleanup_old_positions()
                
                # Print result
                result = {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "age_ms": int(age_ms),
                    "cameras": [cam_id, other_cam_id]
                }
                print(f"      ✅ TRIANGULATION SUCCESS: {json.dumps(result)}")
                successful_triangulations += 1
                
            except Exception as e:
                print(f"      ❌ Triangulation failed: {e}")
                import traceback
                print(f"         {traceback.format_exc()}")
        
        print(f"   📈 Triangulation summary: {successful_triangulations}/{triangulation_attempts} successful")
        
        # Print current system status
        self.print_status_summary()
    
    def calculate_detection_ray(self, cam_id: str, cx: float, cy: float, timestamp: float):
        """
        Calculate and store detection ray for visualization.
        
        Args:
            cam_id: Camera identifier
            cx, cy: Pixel coordinates of detection
            timestamp: Detection timestamp
        """
        if cam_id not in self.poses:
            return
        
        try:
            # Get camera pose
            pose = self.poses[cam_id]
            
            # Calculate ray direction from pixel coordinates
            ray_direction = ray_from_pixel(cx, pose)
            
            # Camera position
            cam_pos = np.array([pose['x'], pose['y']])
            
            # Calculate ray end point (extend ray for visualization)
            ray_length = 8.0  # meters - how far to draw the ray
            ray_end = cam_pos + ray_direction * ray_length
            
            # Store ray for visualization
            self.detection_rays[cam_id] = {
                'start': cam_pos,
                'end': ray_end,
                'timestamp': timestamp,
                'cx': cx,
                'cy': cy
            }
            
            print(f"   📡 Detection ray: {cam_id} from ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}) to ({ray_end[0]:.1f}, {ray_end[1]:.1f})")
            
        except Exception as e:
            print(f"   ❌ Error calculating detection ray for {cam_id}: {e}")
    
    def cleanup_old_detection_rays(self):
        """Remove detection rays older than ray_timeout."""
        current_time = time.time()
        expired_rays = []
        
        for cam_id, ray_data in self.detection_rays.items():
            if (current_time - ray_data['timestamp']) > self.ray_timeout:
                expired_rays.append(cam_id)
        
        for cam_id in expired_rays:
            del self.detection_rays[cam_id]
    
    def get_active_cameras(self) -> set:
        """
        Get set of camera IDs that have sent data within the timeout period.
        
        Returns:
            Set of active camera IDs
        """
        current_time = time.time()
        active_cameras = set()
        
        for cam_id, last_seen in self.camera_last_seen.items():
            if (current_time - last_seen) <= self.camera_timeout:
                active_cameras.add(cam_id)
        
        return active_cameras
    
    def cleanup_inactive_cameras(self):
        """Remove cameras that haven't sent data recently."""
        current_time = time.time()
        inactive_cameras = []
        
        for cam_id, last_seen in self.camera_last_seen.items():
            if (current_time - last_seen) > self.camera_timeout:
                inactive_cameras.append(cam_id)
        
        # Remove inactive cameras
        for cam_id in inactive_cameras:
            if cam_id in self.poses:
                print(f"Camera {cam_id} went inactive, removing from plot")
                del self.poses[cam_id]
                self.plot_needs_update = True
            
            if cam_id in self.camera_last_seen:
                del self.camera_last_seen[cam_id]
            
            if cam_id in self.latest_detections:
                del self.latest_detections[cam_id]
            
            # Also remove detection rays for inactive cameras
            if cam_id in self.detection_rays:
                del self.detection_rays[cam_id]
    
    def triangulate_pair(self, detection1: dict, detection2: dict) -> np.ndarray:
        """
        Triangulate object position from two camera detections.
        
        Args:
            detection1, detection2: Detection packets from different cameras
            
        Returns:
            2D position [x, y] in world coordinates
        """
        # Get camera poses
        pose1 = self.poses[detection1['cam_id']]
        pose2 = self.poses[detection2['cam_id']]
        
        # Convert pixel coordinates to world rays
        ray1 = ray_from_pixel(detection1['cx'], pose1)
        ray2 = ray_from_pixel(detection2['cx'], pose2)
        
        # Camera positions in world coordinates
        pos1 = np.array([pose1['x'], pose1['y']])
        pos2 = np.array([pose2['x'], pose2['y']])
        
        # Triangulate intersection
        position = triangulate_two_rays(pos1, ray1, pos2, ray2)
        
        return position
    
    def cleanup_old_positions(self):
        """Remove position data older than 0.5 seconds. Must be called with lock held."""
        current_time = time.time()
        cutoff_time = current_time - 0.5  # Keep positions for only 0.5 seconds
        
        # Filter out old positions
        valid_indices = [i for i, t in enumerate(self.timestamps) if t > cutoff_time]
        self.positions = [self.positions[i] for i in valid_indices]
        self.timestamps = [self.timestamps[i] for i in valid_indices]
    
    def update_plot(self, frame):
        """Animation callback to update scatter plot with latest positions."""
        if not self.running:
            return self.scatter,
        
        # Check for inactive cameras every 20 frames (~1 second at 50ms intervals)
        if frame % 20 == 0:
            self.cleanup_inactive_cameras()
            self.cleanup_old_detection_rays()
        
        # Check if we need to redraw cameras due to new discoveries or removals
        if self.plot_needs_update:
            self.redraw_cameras()
            self.plot_needs_update = False
        else:
            # Only update detection rays if we're not doing a full redraw
            self.update_detection_rays()
            
        with self._positions_lock:
            # Clean up old positions regularly (every frame)
            self.cleanup_old_positions()
            
            if self.positions:
                # Extract x, y coordinates
                x_coords = [pos[0] for pos in self.positions]
                y_coords = [pos[1] for pos in self.positions]
                
                # Update scatter plot
                self.scatter.set_offsets(np.column_stack([x_coords, y_coords]))
                
                # Auto-adjust axis limits based on data, but include camera positions
                if x_coords and y_coords:
                    # Get camera positions
                    cam_x_coords = [pose['x'] for pose in self.poses.values()]
                    cam_y_coords = [pose['y'] for pose in self.poses.values()]
                    
                    # Combine object and camera coordinates for axis limits
                    all_x = x_coords + cam_x_coords
                    all_y = y_coords + cam_y_coords
                    
                    margin = 1.0
                    x_min, x_max = min(all_x) - margin, max(all_x) + margin
                    y_min, y_max = min(all_y) - margin, max(all_y) + margin
                    
                    # Only update limits if they've changed significantly
                    current_xlim = self.ax.get_xlim()
                    current_ylim = self.ax.get_ylim()
                    
                    if (abs(current_xlim[0] - x_min) > 0.5 or abs(current_xlim[1] - x_max) > 0.5 or
                        abs(current_ylim[0] - y_min) > 0.5 or abs(current_ylim[1] - y_max) > 0.5):
                        self.ax.set_xlim(x_min, x_max)
                        self.ax.set_ylim(y_min, y_max)
                
                # Force canvas refresh only when needed
                if frame % 5 == 0:  # Refresh every 5 frames to reduce CPU usage
                    self.fig.canvas.draw_idle()
            else:
                # No positions - clear the scatter plot
                self.scatter.set_offsets(np.empty((0, 2)))
        
        return self.scatter,
    
    def update_detection_rays(self):
        """Update detection ray visualization on the plot."""
        # Since we're using a clear-and-redraw approach in redraw_cameras(),
        # we only need to draw current detection rays without removing old ones
        
        # Draw current detection rays
        current_time = time.time()
        for cam_id, ray_data in self.detection_rays.items():
            # Check if ray is still valid (not too old)
            ray_age = current_time - ray_data['timestamp']
            if ray_age > self.ray_timeout:
                continue
            
            # Calculate alpha based on age (fade out over time)
            alpha = max(0.1, 1.0 - (ray_age / self.ray_timeout))
            
            # Choose color based on camera ID
            colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange'}
            color = colors.get(cam_id, 'purple')
            
            # Draw detection ray
            start = ray_data['start']
            end = ray_data['end']
            
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                        color=color, linewidth=3, alpha=alpha, 
                        linestyle='--', zorder=4,
                        label=f'Detection Ray {cam_id}')
            
            # Add arrow at the end to show direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Normalize direction
                dx_norm = dx / length
                dy_norm = dy / length
                
                # Arrow properties
                arrow_length = 0.5
                arrow_end_x = end[0] - dx_norm * arrow_length
                arrow_end_y = end[1] - dy_norm * arrow_length
                
                self.ax.annotate('', xy=(end[0], end[1]), 
                               xytext=(arrow_end_x, arrow_end_y),
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             alpha=alpha, lw=2),
                               zorder=5)
            
            # Add detection info text near the camera
            info_x = start[0] + 0.3
            info_y = start[1] + 0.3
            self.ax.text(info_x, info_y, 
                        f'{cam_id}: ({ray_data["cx"]:.0f},{ray_data["cy"]:.0f})',
                        fontsize=8, color=color, alpha=alpha,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.7), zorder=6)
    
    def redraw_cameras(self):
        """Redraw all camera positions and FOVs when cameras are added/updated."""
        # Instead of trying to remove individual artists (which can fail),
        # we'll clear the entire plot and redraw everything
        
        # Save current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Clear the entire plot
        self.ax.clear()
        
        # Restore plot settings
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Object Position Triangulation')
        self.ax.grid(True)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
        # Recreate the scatter plot for object positions
        self.scatter = self.ax.scatter([], [], c='blue', s=100, alpha=1.0, zorder=5, edgecolors='black', linewidth=1)
        
        # Redraw all active cameras
        active_cameras = self.get_active_cameras()
        active_count = 0
        
        for cam_id, pose in self.poses.items():
            if cam_id in active_cameras:
                self.ax.plot(pose['x'], pose['y'], 'rs', markersize=12, label=f'Cam {cam_id}', zorder=10)
                self.draw_camera_fov(pose, cam_id)
                active_count += 1
        
        # Update legend
        self.ax.legend()
        
        print(f"Redrawn plot with {active_count} active cameras (out of {len(self.poses)} total)")
    
    async def run_server(self):
        """Main server loop receiving UDP packets and processing detections."""
        print(f"Fusion server listening on {self.listen_addr}")
        self.running = True
        
        try:
            # Start UDP receiver in background
            async for packet in udp_recv_async(self.listen_addr):
                if not self.running:
                    break
                self.process_detection(packet)
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.running = False
    
    def start_server_thread(self):
        """Start the UDP server in a separate thread."""
        def run_async_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_server())
            except KeyboardInterrupt:
                pass
            finally:
                loop.close()
        
        self.server_thread = threading.Thread(target=run_async_server, daemon=True)
        self.server_thread.start()
    
    def stop(self):
        """Stop the server and cleanup."""
        self.running = False
        if hasattr(self, 'server_thread'):
            self.server_thread.join(timeout=1.0)
    
    def print_status_summary(self):
        """Print a summary of current system status."""
        current_time = time.time()
        active_cameras = self.get_active_cameras()
        
        print(f"\n📊 SYSTEM STATUS:")
        print(f"   🎥 Total cameras known: {len(self.poses)}")
        print(f"   🟢 Active cameras: {len(active_cameras)} {list(active_cameras)}")
        print(f"   📡 Recent detections: {len(self.latest_detections)}")
        print(f"   📡 Active detection rays: {len(self.detection_rays)}")
        
        # Show camera details
        for cam_id, pose in self.poses.items():
            is_active = cam_id in active_cameras
            last_seen = self.camera_last_seen.get(cam_id, 0)
            age = current_time - last_seen if last_seen > 0 else float('inf')
            
            # Check if camera has recent detection ray
            ray_info = ""
            if cam_id in self.detection_rays:
                ray_age = current_time - self.detection_rays[cam_id]['timestamp']
                ray_info = f" | Ray: {ray_age:.1f}s ago"
            
            status = "🟢 ACTIVE" if is_active else f"🔴 INACTIVE ({age:.1f}s ago)"
            
            print(f"      Cam {cam_id}: {status} at ({pose['x']}, {pose['y']}) facing {pose['yaw_deg']}°{ray_info}")
        
        # Show triangulation data
        with self._positions_lock:
            print(f"   📍 Triangulated positions: {len(self.positions)} (last {0.5}s)")
        
        print("=" * 60)


def main():
    """Parse arguments and run fusion server with live plotting."""
    parser = argparse.ArgumentParser(description='Fusion server for object triangulation')
    parser.add_argument('--listen', default='0.0.0.0:9000', help='Listen address (host:port)')
    parser.add_argument('--poses-dir', default='.', help='Directory containing pose JSON files')
    
    args = parser.parse_args()
    
    # Create server
    server = FusionServer(args.listen, args.poses_dir)
    
    # Start server in background thread
    server.start_server_thread()
    
    # Start animation in main thread
    ani = animation.FuncAnimation(
        server.fig, 
        server.update_plot, 
        interval=50,  # Update every 50ms for smooth animation
        blit=False,  # Disable blitting to fix scatter plot visibility
        cache_frame_data=False  # Disable caching to avoid warning
    )
    
    try:
        # Keep the plot responsive and in foreground
        plt.show(block=True)
    except KeyboardInterrupt:
        print("Server stopped")
    finally:
        server.stop()
        plt.close('all')


if __name__ == "__main__":
    main() 