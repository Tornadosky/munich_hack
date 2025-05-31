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

## 🏷️ **AprilTag-Based Calibration** (Professional Grade)

### **Overview**

The system supports automatic camera pose estimation using AprilTags, providing professional-grade accuracy without manual measurements. AprilTags are robust fiducial markers that enable precise 6DOF pose estimation.

### **How It Works**

1. **AprilTag Placement**: Place AprilTags at known world coordinate positions
2. **Automatic Detection**: Cameras detect AprilTags and calculate their position relative to them
3. **Real-time Updates**: Camera pose updates continuously as tags are detected
4. **High Accuracy**: Uses OpenCV's `solvePnP` function for precise pose estimation

### **Quick Start with AprilTags**

#### Method 1: One-time Calibration Session
```bash
# 1. Generate AprilTag configuration and printable tags
python generate_apriltags.py --config apriltag_config.json

# 2. Print the generated apriltags.png file (at 100% scale, no fit-to-page)

# 3. Place AprilTag ID 0 at position (1, 0) and AprilTag ID 1 at position (-1, 0)

# 4. Run calibration session
python apriltag_pose_calibration.py --camera-id A --config apriltag_config.json

# 5. Use the camera system
python fusion_server.py --listen 0.0.0.0:9000
python cam_client.py --pose pose_A.json --server localhost:9000 --target bottle
```

#### Method 2: Real-time Pose Updates
```bash
# Skip calibration session, let camera auto-update its pose in real-time
python cam_client.py --pose pose_A.json --server localhost:9000 --target bottle \
    --enable-apriltag-pose --apriltag-config apriltag_config.json
```

### **AprilTag Configuration**

The system uses a configuration file (`apriltag_config.json`) to define AprilTag positions:

```json
{
    "apriltags": [
        {
            "id": 0,
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "size": 0.1,
            "description": "AprilTag at position (1, 0)"
        },
        {
            "id": 1,
            "x": -1.0,
            "y": 0.0,
            "z": 0.0,
            "size": 0.1,
            "description": "AprilTag at position (-1, 0)"
        }
    ],
    "tag_family": "tag36h11",
    "default_size": 0.1,
    "camera_intrinsics": {
        "focal_length": 500.0,
        "distortion_coeffs": [0.1, -0.2, 0, 0, 0]
    },
    "pose_estimation": {
        "max_reproj_error": 5.0,
        "min_detection_confidence": 0.3
    }
}
```

### **AprilTag Commands**

```bash
# Generate tags from config file
python generate_apriltags.py --config apriltag_config.json

# Generate specific tag IDs
python generate_apriltags.py --tag-ids 0 1 2 --physical-size 0.1

# Run calibration session
python apriltag_pose_calibration.py --camera-id A --config apriltag_config.json

# Use real-time pose updates
python cam_client.py --pose pose_A.json --server localhost:9000 --target bottle \
    --enable-apriltag-pose --apriltag-config apriltag_config.json
```

### **AprilTag Coordinate System**

- **Tag Origin**: Center of the tag
- **X-axis**: Left edge to right edge of tag
- **Y-axis**: Bottom edge to top edge of tag  
- **Z-axis**: Perpendicular to tag surface (outward)
- **Tag corners**: Bottom-left, bottom-right, top-right, top-left

### **Accuracy and Performance**

- **Position accuracy**: ±2-5mm (depends on tag size, distance, and lighting)
- **Angular accuracy**: ±1-2 degrees
- **Detection range**: 0.5m to 5m (for 10cm tags)
- **Update rate**: 1-10 Hz (configurable)
- **Requirements**: Good lighting, stable tag placement, clear line of sight

### **Troubleshooting AprilTags**

**Common Issues:**
- **No tags detected**: Check lighting, tag print quality, camera focus
- **Poor accuracy**: Increase tag size, reduce distance, improve lighting
- **Intermittent detection**: Stabilize tag placement, check for reflections
- **Wrong pose**: Verify tag positions in config file match physical placement

**Debug Commands:**
```bash
# Test AprilTag detection
python apriltag_pose_calibration.py --camera-id TEST --samples 1

# Generate tags with different sizes
python generate_apriltags.py --tag-ids 1 2 --physical-size 0.15  # 15cm tags

# Check AprilTag installation
python -c "import apriltag; print('AprilTag OK')"
```

### **Status Logging and Monitoring**

The AprilTag system includes comprehensive status logging to help with debugging and monitoring:

#### **Calibration Session Logs**
During calibration (`apriltag_pose_calibration.py`), you'll see:
```
📊 APRILTAG DETECTION STATUS:
   Frames processed: 150
   Detection rate: 73.3% (110/150)
   Total tags detected: 165
   Detection time: 12.3ms
   Current frame: 2 tags detected
      Tag 1: conf=0.847, area=2145px², aspect=0.98, pos=(1.0, 0.0)
      Tag 2: conf=0.692, area=1834px², aspect=1.02, pos=(-1.0, 0.0)
   Tag detection history:
      Tag 1: 89/95 (93.7% success)
      Tag 2: 76/89 (85.4% success)
```

#### **Real-time Pose Update Logs**
When using `--enable-apriltag-pose`, you'll see detailed updates:
```
🏷️ APRILTAG POSE UPDATE #5 for Camera A:
   📍 CAMERA POSITION:
      Old: (2.145, 1.832) m
      New: (2.147, 1.829) m
      Change: 0.004m
   🧭 CAMERA ORIENTATION:
      Old: 45.2°
      New: 45.1°
      Change: 0.1°
   🏷️ APRILTAG DETECTION:
      Reference Tag ID: 1
      Tag Position: [1.0, 0.0, 0.0]
      Distance to Tag: 2.341m
      Detection Confidence: 0.847
      Reprojection Error: 1.23px
      Estimated FOV: 68.5°
   📊 DETECTION STATISTICS:
      Detection Rate: 73.3%
      Pose Success Rate: 89.2%
      Total Tags Detected: 165
      Last Detection: 0.1 seconds ago
   📡 SERVER TRANSMISSION:
      Updated pose will be sent with next detection packet
      Server: localhost:9000
```

#### **Detection Failure Diagnostics**
When AprilTags aren't detected, the system logs diagnostic information:
```
⚠️ AprilTag pose update failed for Camera A:
   📊 Detection Statistics:
      Frames processed: 450
      Detection rate: 23.1%
      Successful poses: 12
      Failed poses: 8
   🏷️ Tag Detection Status:
      Tag 1: 45/67 (67.2% success)
      Tag 2: 23/45 (51.1% success)
```

#### **Performance Monitoring**
The system tracks and displays:
- **Detection Rate**: Percentage of frames with detected tags
- **Pose Success Rate**: Percentage of detections leading to successful pose estimation
- **Reprojection Error**: Quality metric for pose accuracy (lower is better)
- **Detection Time**: Time taken for AprilTag detection per frame
- **Per-Tag Statistics**: Individual success rates for each configured tag

#### **Interpreting the Logs**

**Good Performance Indicators:**
- Detection rate > 70%
- Pose success rate > 85%
- Reprojection error < 3.0px
- Detection time < 20ms

**Troubleshooting Low Performance:**
- Detection rate < 50% → Check lighting, tag visibility, print quality
- Pose success rate < 70% → Check tag placement, camera focus, tag size
- High reprojection error > 5px → Verify tag positions in config file
- Detection time > 50ms → Consider using smaller images or faster hardware

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