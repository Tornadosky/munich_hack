"""
Math and UDP utility functions for 2D object triangulation system.
Provides ray conversion, triangulation, and network communication helpers.
"""

import numpy as np
import math
import json
import socket
import asyncio
from typing import Dict, Any, Tuple, AsyncIterator


def ray_from_pixel(cx_px: float, pose_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convert pixel x-coordinate to unit ray vector in world coordinates.
    
    Args:
        cx_px: Pixel x-coordinate of detection center
        pose_dict: Camera pose containing x, y, yaw_deg, fov_deg, img_w
        
    Returns:
        2D unit vector (cos, sin) representing ray direction
    """
    img_w = pose_dict['img_w']
    fov_deg = pose_dict['fov_deg']
    yaw_deg = pose_dict['yaw_deg']
    
    # Convert pixel offset to angle offset from camera center
    rel_angle_deg = (cx_px - img_w/2) / img_w * fov_deg
    
    # Add to camera yaw to get world bearing
    bearing_deg = yaw_deg + rel_angle_deg
    bearing_rad = math.radians(bearing_deg)
    
    # Return unit vector in world coordinates
    return np.array([math.cos(bearing_rad), math.sin(bearing_rad)])


def triangulate_two_rays(p1: np.ndarray, r1: np.ndarray, 
                        p2: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """
    Find least-squares intersection of two 2D rays using closed-form solution.
    
    Args:
        p1, p2: Ray start points as 2D arrays [x, y]
        r1, r2: Ray direction vectors as 2D arrays [dx, dy]
        
    Returns:
        2D point [x, y] of closest intersection
        
    Raises:
        ValueError: If rays are parallel (determinant near zero)
    """
    # Solve system: p1 + t1*r1 = p2 + t2*r2
    # Rearrange to: t1*r1 - t2*r2 = p2 - p1
    
    # Build coefficient matrix A = [r1, -r2]
    A = np.column_stack([r1, -r2])
    b = p2 - p1
    
    # Check for parallel rays
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        raise ValueError("Rays are parallel or nearly parallel")
    
    # Solve for parameters t1, t2
    t = np.linalg.solve(A, b)
    t1 = t[0]
    
    # Return intersection point
    return p1 + t1 * r1


def udp_send(payload: Dict[str, Any], addr: Tuple[str, int]) -> None:
    """
    Send JSON payload via UDP to specified address.
    
    Args:
        payload: Dictionary to send as JSON
        addr: (host, port) tuple for destination
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        message = json.dumps(payload).encode('utf-8')
        sock.sendto(message, addr)
        sock.close()
    except Exception as e:
        print(f"UDP send error: {e}")


async def udp_recv_async(listen_addr: Tuple[str, int]) -> AsyncIterator[Dict[str, Any]]:
    """
    Async generator yielding JSON packets received on UDP socket.
    
    Args:
        listen_addr: (host, port) tuple to bind to
        
    Yields:
        Parsed JSON dictionaries from incoming packets
    """
    class UDPProtocol(asyncio.DatagramProtocol):
        def __init__(self):
            self.queue = asyncio.Queue()
            
        def datagram_received(self, data, addr):
            try:
                payload = json.loads(data.decode('utf-8'))
                self.queue.put_nowait(payload)
            except Exception as e:
                print(f"UDP receive error: {e}")
    
    # Create UDP endpoint
    loop = asyncio.get_event_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        UDPProtocol, local_addr=listen_addr
    )
    
    try:
        while True:
            payload = await protocol.queue.get()
            yield payload
    finally:
        transport.close() 