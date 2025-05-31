# Multi-Camera Object Triangulation System

A minimal-friction prototype for real-time object localization using two laptops with webcams. The system uses YOLOv8 object detection and UDP communication to triangulate target positions on a 2D map.

## Installation

```bash
pip install -r requirements.txt
```

## 🚀 **Super Simple Setup** (Recommended)

### **Automatic Calibration - Zero Measurements!**

```bash
# 1. Place both laptops in room (2+ meters apart)
# 2. Run automatic calibration
python auto_calibration.py
```

**What it does:**
- Opens both camera feeds simultaneously
- You move a bottle to 3-4 different positions
- Press SPACE when both cameras see the bottle
- System automatically calculates camera positions
- **No tape measure, no manual measurements!**

**Then run your system:**
```bash
# Server laptop:
python fusion_server.py --listen 0.0.0.0:9000

# Both laptops:
python cam_client.py --pose pose_A.json --server [SERVER_IP]:9000 --target bottle
python cam_client.py --pose pose_B.json --server [SERVER_IP]:9000 --target bottle
```

That's it! **Total setup time: 5 minutes.**

---

## 🎯 Manual Setup (Alternative)

If you want precise control or automatic calibration doesn't work:

### Step 1: Physical Setup
- Place your 2 laptops in different corners/sides of the room
- **Important**: Keep them 2+ meters apart for good triangulation
- Point cameras toward the center area where you'll detect bottles
- Choose one laptop as the "server" (runs fusion_server.py)

### Step 2: Test YOLO Detection
```bash
# Test on each laptop to ensure detection works
python smooth_yolo_test.py --target bottle --confidence 0.3
```
Make sure both laptops can detect bottles reliably before proceeding.

### Step 3: Measure Camera Positions
```bash
# Run on ONE laptop (doesn't matter which)
python camera_calibration.py --mode measure
```
This will guide you through:
- Choose a room corner as origin (0,0)
- Measure each laptop's position with tape measure
- Determine orientations (which way cameras point)
- Add reference points for testing

### Step 4: Calibrate Field of View
```bash
# Run on EACH laptop individually
python camera_calibration.py --mode fov
```
- Use a bottle/book of known width (e.g., 0.07m for bottle)
- Place at known distance (e.g., 1.5m from camera)
- Click edges to measure pixels
- Get accurate FOV for that camera

### Step 5: Generate Pose Files
```bash
# Run after both FOV calibrations are done
python camera_calibration.py --mode generate
```
Creates `pose_A.json` and `pose_B.json` with accurate measurements.

### Step 6: Test Calibration
```bash
# Verify accuracy before running full system
python camera_calibration.py --mode test
```
- Place bottle at known reference point
- Check triangulation error (should be < 30cm)

### Step 7: Run Triangulation System
```bash
# Server laptop (runs fusion and displays results)
python fusion_server.py --listen 0.0.0.0:9000

# Client laptops (run detection and send data)
# Laptop A:
python cam_client.py --pose pose_A.json --server 192.168.1.X:9000 --target bottle

# Laptop B:
python cam_client.py --pose pose_B.json --server 192.168.1.X:9000 --target bottle
```
Replace `192.168.1.X` with the server laptop's IP address.

## 📏 Key Measurements Explained

**Position (x, y)**: 
- Choose room corner as origin (0, 0)
- Measure with tape from corner to CENTER of laptop screen
- X = horizontal distance, Y = forward distance

**Orientation (yaw_deg)**:
- 0° = East (positive X direction)
- 90° = North (positive Y direction)
- 180° = West, 270° = South
- Use a compass or visual reference

**Field of View (fov_deg)**:
- Calibrated automatically using known object
- Don't guess - measure precisely for accuracy

## 💡 Pro Tips

**For Best Results:**
- Keep laptops 2+ meters apart
- Point cameras toward room center
- Ensure good lighting for bottle detection
- Use lower confidence: `--confidence 0.3`

**Troubleshooting:**
- Can't detect bottle? → Test with `python smooth_yolo_test.py --target bottle`
- Auto-calibration fails? → Use manual method above
- Network issues? → Check IP addresses with `ipconfig`

## Quick Demo (No Webcams Required)

```bash
# Terminal 1: Start fusion server
python fusion_server.py

# Terminal 2: Run fake camera demo
python demo_fake.py
```

## Example Pose Files

After calibration, your files should look like:

```json
# pose_A.json
{
  "cam_id": "A",
  "x": 0.0,
  "y": 0.0,
  "yaw_deg": 45.0,
  "fov_deg": 70.0,
  "img_w": 640,
  "img_h": 480
}
```

```json
# pose_B.json
{
  "cam_id": "B", 
  "x": 3.5,
  "y": 2.0,
  "yaw_deg": 225.0,
  "fov_deg": 70.0,
  "img_w": 640,
  "img_h": 480
}
```

## Testing

```bash
pytest tests/
```

The system triangulates object positions when detections from two cameras arrive within 0.5 seconds, displaying results in real-time on a matplotlib scatter plot. 