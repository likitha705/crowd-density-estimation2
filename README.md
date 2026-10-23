# Crowd Density Estimation — Streamlit + pretrained CNN

This app uses a pretrained YOLO11n model to detect visible people in a crowd image, count detections, display bounding boxes, and show a detection-concentration heatmap overlay.

## Files
- `crowd_density_app.py` — Streamlit app
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run crowd_density_app.py
```

The first run downloads `yolo11n.pt` automatically, so internet access is required.

## Deploy to Streamlit Community Cloud
1. Upload both files to the root of your GitHub repository.
2. In Streamlit Community Cloud, select your repository and branch.
3. Set **Main file path** to `crowd_density_app.py`.
4. Click **Deploy**.

## Important model note
This version uses pretrained YOLO11n person detection (COCO class `person`). It is a real pretrained deep-learning model and gives an estimated number of detected people. It is **not** a trained density-map regression model such as CSRNet. The displayed heatmap is generated from detected person locations and is not a true regression density map. In very dense or occluded crowds, person detection may miss people, so counts are estimates, not guaranteed exact values. The low/medium/high thresholds (under 10, 10–25, above 25) are simple demo thresholds and should be adjusted for the intended scene.
