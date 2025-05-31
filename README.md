# Multi-Camera Object Triangulation System

A minimal-friction prototype for real-time object localization using two laptops with webcams. The system uses YOLOv8 object detection and UDP communication to triangulate target positions on a 2D map.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Demo (No Webcams Required)

```bash
# Terminal 1: Start fusion server
python fusion_server.py

# Terminal 2: Run fake camera demo
python demo_fake.py
```

## Real Camera Setup

1. **Create camera pose files** on each laptop:

```json
# pose_A.json (Camera A)
{
  "cam_id": "A",
  "x": 0.0,
  "y": 0.0,
  "yaw_deg": 90.0,
  "fov_deg": 70.0,
  "img_w": 640,
  "img_h": 480
}
```

```json
# pose_B.json (Camera B) 
{
  "cam_id": "B",
  "x": 4.0,
  "y": 0.0,
  "yaw_deg": 270.0,
  "fov_deg": 70.0,
  "img_w": 640,
  "img_h": 480
}
```

2. **Start fusion server** on one laptop:
```bash
python fusion_server.py --listen 0.0.0.0:9000
```

3. **Start camera clients** on each laptop:
```bash
# Laptop A
python cam_client.py --pose pose_A.json --server 192.168.1.90:9000 --target bottle

# Laptop B  
python cam_client.py --pose pose_B.json --server 192.168.1.90:9000 --target bottle
```

## Testing

```bash
pytest tests/
```

The system triangulates object positions when detections from two cameras arrive within 0.5 seconds, displaying results in real-time on a matplotlib scatter plot. 