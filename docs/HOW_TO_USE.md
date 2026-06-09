# How to Use the GNSS-Free Navigation App

## Step 1: Start the Application

```bash
python src/vns/app/desktop_app.py
```

## Step 2: Enter Coordinates

- **Start Point**: Enter latitude and longitude of starting location
- **End Point**: Enter latitude and longitude of destination
- Click **Plot Points** — green dot is start, red dot is end

## Step 3: Open Google Earth Pro

- Click **Open in Google Earth Pro**
- App creates a KML file and opens Google Earth at your coordinates
- Press **U** in Google Earth for top-down view
- Press **N** for north-up orientation

## Step 4: Draw Your Area

- In Google Earth, use **Add Polygon** tool
- Draw around your mission area
- Save as KML to the `data/kml/` folder
- App detects it automatically within 2 seconds

## Step 5: Load Satellite Image

- Click **Load Satellite Image**
- Select your satellite image file
- Polygon overlay appears on image

## Step 6: Run Path Planning

- Click **Set Start/Goal on Map** then click on the image
- Click **Run A* Path Planning**
- Purple path appears showing optimal route
- Waypoints are generated automatically

## Step 7: Export Waypoints

- Click **Export CSV** or **Export JSON**
- Files saved to `data/waypoints/`

---

## Coordinate Format

Use decimal degrees format:
- Latitude: `33.7470`
- Longitude: `73.1370`

QAU Campus center: `33.7470, 73.1370`