"""
Fusion server for multi-camera object triangulation.
Receives detection packets via UDP and triangulates object positions using camera poses.
"""

import json
import time
import argparse
import asyncio
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
            pose_dir: Directory to search for pose_*.json files
        """
        # Parse listen address
        host, port = listen_addr.split(':')
        self.listen_addr = (host, int(port))
        
        # Load all camera poses from JSON files
        self.poses = {}
        pose_files = glob(f"{pose_dir}/pose_*.json")
        
        for pose_file in pose_files:
            with open(pose_file, 'r') as f:
                pose = json.load(f)
                self.poses[pose['cam_id']] = pose
                print(f"Loaded pose for camera {pose['cam_id']}")
        
        if len(self.poses) < 2:
            print("Warning: Need at least 2 camera poses for triangulation")
        
        # Storage for latest detections per camera
        self.latest_detections = {}
        self.time_window = 0.5  # seconds
        
        # Triangulated positions for plotting
        self.positions = []
        self.timestamps = []
        
        # Setup matplotlib for live plotting
        self.setup_plot()
    
    def setup_plot(self):
        """Initialize matplotlib figure for real-time position plotting."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('Object Position Triangulation')
        self.ax.grid(True)
        
        # Plot camera positions
        for cam_id, pose in self.poses.items():
            self.ax.plot(pose['x'], pose['y'], 'rs', markersize=10, label=f'Cam {cam_id}')
        
        # Initialize empty scatter plot for object positions
        self.scatter = self.ax.scatter([], [], c='blue', s=50, alpha=0.7)
        
        # Set initial axis limits
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.legend()
    
    def process_detection(self, packet: dict):
        """
        Process incoming detection packet and attempt triangulation.
        
        Args:
            packet: Detection packet with cam_id, cx, cy, timestamp
        """
        cam_id = packet['cam_id']
        timestamp = packet['timestamp']
        
        # Store latest detection for this camera
        self.latest_detections[cam_id] = packet
        
        # Try triangulation with other cameras
        current_time = time.time()
        
        for other_cam_id, other_detection in self.latest_detections.items():
            if other_cam_id == cam_id:
                continue
            
            # Check if detections are within time window
            time_diff = abs(timestamp - other_detection['timestamp'])
            if time_diff > self.time_window:
                continue
            
            # Perform triangulation
            try:
                position = self.triangulate_pair(packet, other_detection)
                age_ms = (current_time - min(timestamp, other_detection['timestamp'])) * 1000
                
                # Store result
                self.positions.append(position)
                self.timestamps.append(current_time)
                
                # Print result
                result = {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "age_ms": int(age_ms)
                }
                print(json.dumps(result))
                
                # Keep only recent positions for plotting
                self.cleanup_old_positions()
                
            except Exception as e:
                print(f"Triangulation failed: {e}")
    
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
        """Remove position data older than 10 seconds."""
        current_time = time.time()
        cutoff_time = current_time - 10.0
        
        # Filter out old positions
        valid_indices = [i for i, t in enumerate(self.timestamps) if t > cutoff_time]
        self.positions = [self.positions[i] for i in valid_indices]
        self.timestamps = [self.timestamps[i] for i in valid_indices]
    
    def update_plot(self, frame):
        """Animation callback to update scatter plot with latest positions."""
        if self.positions:
            # Extract x, y coordinates
            x_coords = [pos[0] for pos in self.positions]
            y_coords = [pos[1] for pos in self.positions]
            
            # Update scatter plot
            self.scatter.set_offsets(np.column_stack([x_coords, y_coords]))
            
            # Auto-adjust axis limits based on data
            if x_coords and y_coords:
                margin = 1.0
                x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
                y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
                
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_min, y_max)
        
        return self.scatter,
    
    async def run_server(self):
        """Main server loop receiving UDP packets and processing detections."""
        print(f"Fusion server listening on {self.listen_addr}")
        
        # Start UDP receiver in background
        async for packet in udp_recv_async(self.listen_addr):
            self.process_detection(packet)


async def main():
    """Parse arguments and run fusion server with live plotting."""
    parser = argparse.ArgumentParser(description='Fusion server for object triangulation')
    parser.add_argument('--listen', default='0.0.0.0:9000', help='Listen address (host:port)')
    parser.add_argument('--poses-dir', default='.', help='Directory containing pose JSON files')
    
    args = parser.parse_args()
    
    # Create server
    server = FusionServer(args.listen, args.poses_dir)
    
    # Start animation (5 Hz update rate)
    ani = animation.FuncAnimation(server.fig, server.update_plot, interval=200, blit=True)
    
    # Run server and show plot concurrently
    plt.ion()
    plt.show()
    
    try:
        await server.run_server()
    except KeyboardInterrupt:
        print("Server stopped")
    finally:
        plt.close()


if __name__ == "__main__":
    asyncio.run(main()) 