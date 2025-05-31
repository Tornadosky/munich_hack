"""
Demo script generating synthetic object detections for testing.
Simulates two cameras tracking a moving object and sends UDP packets to fusion server.
"""

import asyncio
import json
import time
import math
import numpy as np
from utils import udp_send, ray_from_pixel


class FakeCamera:
    """Simulates a camera sending synthetic detection packets."""
    
    def __init__(self, pose_dict: dict, server_addr: tuple, fps: float = 2.0):
        """
        Initialize fake camera with pose and server address.
        
        Args:
            pose_dict: Camera pose dictionary
            server_addr: (host, port) tuple for server
            fps: Simulation frame rate
        """
        self.pose = pose_dict
        self.server_addr = server_addr
        self.fps = fps
        self.interval = 1.0 / fps
    
    def world_to_pixel(self, world_x: float, world_y: float) -> tuple:
        """
        Convert world coordinates to pixel coordinates for this camera.
        
        Args:
            world_x, world_y: Object position in world coordinates
            
        Returns:
            (cx, cy) pixel coordinates, or None if not in view
        """
        # Vector from camera to object
        cam_x, cam_y = self.pose['x'], self.pose['y']
        obj_vec = np.array([world_x - cam_x, world_y - cam_y])
        
        # Camera direction vector
        yaw_rad = math.radians(self.pose['yaw_deg'])
        cam_dir = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
        
        # Calculate bearing to object
        obj_bearing = math.atan2(obj_vec[1], obj_vec[0])
        cam_bearing = math.atan2(cam_dir[1], cam_dir[0])
        
        # Relative angle from camera center
        rel_angle = obj_bearing - cam_bearing
        
        # Normalize to [-pi, pi]
        while rel_angle > math.pi:
            rel_angle -= 2 * math.pi
        while rel_angle < -math.pi:
            rel_angle += 2 * math.pi
        
        # Check if object is within field of view
        fov_rad = math.radians(self.pose['fov_deg'])
        if abs(rel_angle) > fov_rad / 2:
            return None  # Object not in view
        
        # Convert to pixel coordinates
        rel_angle_deg = math.degrees(rel_angle)
        img_w = self.pose['img_w']
        
        cx = img_w / 2 + (rel_angle_deg / self.pose['fov_deg']) * img_w
        cy = self.pose['img_h'] / 2  # Assume object at center height
        
        # Check if pixel is within image bounds
        if 0 <= cx <= img_w and 0 <= cy <= self.pose['img_h']:
            return (cx, cy)
        
        return None
    
    def send_detection(self, cx: float, cy: float):
        """
        Send synthetic detection packet to server.
        
        Args:
            cx, cy: Pixel coordinates of detected object
        """
        packet = {
            "cam_id": self.pose['cam_id'],
            "cx": cx,
            "cy": cy,
            "timestamp": time.time()
        }
        
        udp_send(packet, self.server_addr)
        print(f"Fake cam {self.pose['cam_id']}: sent detection at ({cx:.1f}, {cy:.1f})")
    
    async def track_object(self, object_path):
        """
        Simulate tracking an object following the given path.
        
        Args:
            object_path: List of (x, y, timestamp) tuples
        """
        for world_x, world_y, _ in object_path:
            # Convert world position to pixel coordinates
            pixel_coords = self.world_to_pixel(world_x, world_y)
            
            if pixel_coords is not None:
                cx, cy = pixel_coords
                self.send_detection(cx, cy)
            else:
                print(f"Fake cam {self.pose['cam_id']}: object not in view at ({world_x:.1f}, {world_y:.1f})")
            
            # Wait for next frame
            await asyncio.sleep(self.interval)


def generate_object_path(duration: float = 20.0, fps: float = 2.0):
    """
    Generate a synthetic path for a moving object.
    
    Args:
        duration: Path duration in seconds
        fps: Path sampling rate
        
    Returns:
        List of (x, y, timestamp) tuples
    """
    path = []
    start_time = time.time()
    dt = 1.0 / fps
    
    # Create circular path with radius 2m centered at (2, 0)
    center_x, center_y = 2.0, 0.0
    radius = 2.0
    angular_speed = 2 * math.pi / 10.0  # Complete circle in 10 seconds
    
    t = 0.0
    while t < duration:
        # Parametric circle
        x = center_x + radius * math.cos(angular_speed * t)
        y = center_y + radius * math.sin(angular_speed * t)
        
        path.append((x, y, start_time + t))
        t += dt
    
    return path


async def main():
    """Run fake camera demo with two simulated cameras."""
    # Server address
    server_addr = ('localhost', 9000)
    
    # Define two camera poses
    pose_a = {
        "cam_id": "A",
        "x": 0.0,
        "y": 0.0,
        "yaw_deg": 45.0,    # Northeast
        "fov_deg": 70.0,
        "img_w": 640,
        "img_h": 480
    }
    
    pose_b = {
        "cam_id": "B", 
        "x": 4.0,
        "y": 0.0,
        "yaw_deg": 135.0,   # Northwest
        "fov_deg": 70.0,
        "img_w": 640,
        "img_h": 480
    }
    
    # Save pose files for fusion server
    with open('pose_A.json', 'w') as f:
        json.dump(pose_a, f, indent=2)
    
    with open('pose_B.json', 'w') as f:
        json.dump(pose_b, f, indent=2)
    
    print("Created pose files: pose_A.json, pose_B.json")
    
    # Create fake cameras
    fake_cam_a = FakeCamera(pose_a, server_addr, fps=2.0)
    fake_cam_b = FakeCamera(pose_b, server_addr, fps=2.0)
    
    # Generate object path
    object_path = generate_object_path(duration=30.0, fps=2.0)
    
    print(f"Starting demo with {len(object_path)} path points")
    print("Make sure fusion_server.py is running!")
    print("Object will move in a circle between the two cameras")
    
    # Run both cameras concurrently
    await asyncio.gather(
        fake_cam_a.track_object(object_path),
        fake_cam_b.track_object(object_path)
    )
    
    print("Demo completed")


if __name__ == "__main__":
    asyncio.run(main()) 