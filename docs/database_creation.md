# Reference Database Creation Guide

This guide walks you through building, capturing, and validating a geotagged reference database (`.vnsdb`) for the Visual Navigation System.

---

## 1. Flight Planning for Reference Image Capture

To ensure robust visual matching, reference images must provide continuous coverage with high spatial overlap.

- **Overlap:** Ensure at least **60% forward overlap** and **40% side overlap** between image capture points.
- **Altitude:** Capture images at the same target altitude as your intended mission (typically 30m to 60m AGL).
- **Lighting:** Capture under uniform, diffuse lighting (mid-day or overcast) to prevent long shadows from skewing feature descriptors.
- **Pattern:** Use a lawnmower flight grid pattern for systematic coverage.

---

## 2. Capturing Geotagged Images

### Simulation Capture
To capture reference images from the Gazebo simulation environment using our built-in synthetic image generator:
```bash
# Capture synthetic reference images based on QAU Campus metadata
vns database capture --synthetic --config simulation/database/qau_reference_metadata.yaml --output simulation/database/images
```

### Live Hardware Capture
For a real drone flight, collect high-resolution JPG images with geotag coordinates (latitude, longitude, altitude, and yaw/heading) written into a database index YAML file:
```yaml
database:
  name: "QAU Campus Reference Database"
  version: "1.0.0"
  location: "Islamabad, Pakistan"
  bounds:
    min_latitude: 33.740
    max_latitude: 33.755
    min_longitude: 73.130
    max_longitude: 73.145
images:
  - id: "qau_admin_01"
    filepath: "simulation/database/images/qau_admin_01.jpg"
    latitude: 33.7470
    longitude: 73.1370
    altitude: 580.0
    heading: 0.0
```

---

## 3. Indexing and Building the Feature Database

Once the images are collected and indexed, process them with our CLI batch indexing tool to extract ORB descriptors and package them into a binary `.vnsdb` format:
```bash
vns database build --input simulation/database/images/database_index.yaml --output simulation/database/qau_campus.vnsdb
```

---

## 4. Validating and Inspecting the Reference Database

Before deployment, verify the integrity, feature count, and boundaries of your reference database using the inspection tool:
```bash
vns database inspect simulation/database/qau_campus.vnsdb
```

### Coverage Audit
A healthy reference database should have:
- At least **100+ keypoints** extracted per image.
- Geographic bounds matching the target area.
- No duplicate entries.
