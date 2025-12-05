import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os

# --- 0. Configuration ---
st.set_page_config(page_title="Smart Vehicle Detection and Toll Management", layout="wide", page_icon="🚗")

# Define path to models
MODEL_PATHS = {
    "Best Model (best.pt)": "best.pt",
    "Epoch 150 Model (best150.pt)": "best150.pt",
    "Standard YOLOv8n": "yolov8n.pt"  # Fallback
}

UNAUTHORIZED_DEFAULTS = {"rickshaw", "cng", "van"}

DEFAULT_TOLL_RATES = {
    "_default": 100,
    "car": 150,
    "bus": 250,
    "truck": 300,
    "motorcycle": 60,
    "rickshaw": 0,
    "cng": 0,
    "van": 120,
}

# --- 1. Caching Logic ---
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, e

# --- 2. Tracking & Counting Logic (The Core Update) ---
def process_frame_with_tracking(model, frame, conf_thresh, line_position, unauthorized_lookup, counted_ids):
    """
    Runs tracking, draws the tripwire, checks for crossing, and handles alerts.
    """
    # 1. Run Tracking (persist=True is vital for video)
    results = model.track(frame, persist=True, conf=conf_thresh, verbose=False)
    
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]
    
    # Define Line Coordinates (Horizontal line)
    line_y = int(height * line_position)
    cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2) # Yellow Tripwire
    cv2.putText(annotated_frame, "TOLL LINE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Variables for this specific frame
    frame_events = {
        "authorized_count": 0,
        "unauthorized_count": 0,
        "alert_triggered": False,
        "authorized_crossings": [],
        "unauthorized_crossings": [],
    }

    if results[0].boxes.id is not None:
        # Get boxes, IDs, and classes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()

        for box, track_id, class_id in zip(boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[class_id]
            class_name_lower = class_name.lower()
            
            # Calculate Center Point
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Determine Object Status
            is_unauthorized = class_name_lower in unauthorized_lookup

            color = (0, 0, 255) if is_unauthorized else (0, 255, 0) # Red for unauthorized, Green for safe

            # Draw Bounding Box & Center
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_frame, (cx, cy), 4, color, -1)
            cv2.putText(annotated_frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- CROSSING LOGIC ---
            # Check if object is within a small buffer of the line AND hasn't been counted yet
            offset = 10 # Buffer zone in pixels
            if (line_y - offset) < cy < (line_y + offset):
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    
                    if is_unauthorized:
                        frame_events["unauthorized_count"] += 1
                        frame_events["unauthorized_crossings"].append(f"ID:{track_id} {class_name}")
                        frame_events["alert_triggered"] = True
                    else:
                        frame_events["authorized_count"] += 1
                        frame_events["authorized_crossings"].append(class_name)

    # Visual Alert on Screen if Unauthorized detected
    if frame_events["alert_triggered"] and frame_events["unauthorized_crossings"]:
        # Build alert text with ID and name for each unauthorized crossing
        alert_lines = ["UNAUTHORIZED:"]
        for class_name in frame_events["unauthorized_crossings"]:
            alert_lines.append(f"{class_name}")
        alert_text = " | ".join(alert_lines)
        
        # Position at top-right with padding
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = width - text_size[0] - 20
        text_y = 30
        
        # Background rectangle for better visibility
        cv2.rectangle(annotated_frame, (text_x - 10, text_y - 25), 
                     (width - 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(annotated_frame, alert_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return annotated_frame, frame_events

# --- 3. Video Processing Helper ---
def process_video_stream(model, cap, conf_threshold, line_pos, unauthorized_lookup, toll_rates):
    
    # UI Layout for Real-time Stats
    st.markdown("### 📊 Real-Time Traffic Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        metric_total = st.empty()
    with col2:
        metric_auth = st.empty()
    with col3:
        metric_unauth = st.empty()
    with col4:
        metric_rev = st.empty()

    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    stop_button = st.sidebar.button("⏹ Stop Stream", key="stop_btn")

    # Session State for Cumulative Counting
    total_vehicles = 0
    auth_vehicles = 0
    unauth_vehicles = 0
    total_revenue = 0
    
    # Set to keep track of vehicle IDs that have already crossed
    counted_ids = set()

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Streamlit likes RGB
        
        # Pass data to our new tracking function
        annotated_frame, events = process_frame_with_tracking(
            model, 
            rgb_frame, 
            conf_threshold, 
            line_pos, 
            unauthorized_lookup, 
            counted_ids
        )

        # Update Cumulative Stats
        if events["authorized_crossings"]:
            auth_increment = len(events["authorized_crossings"])
            auth_vehicles += auth_increment
            total_vehicles += auth_increment
            for class_entry in events["authorized_crossings"]:
                # Extract class name if it contains ID prefix
                if "ID:" in class_entry:
                    class_name = class_entry.split(" ", 1)[1]
                else:
                    class_name = class_entry
                total_revenue += toll_rates.get(class_name.lower(), toll_rates.get("_default", 0))
        
        if events["unauthorized_crossings"]:
            unauth_increment = len(events["unauthorized_crossings"])
            unauth_vehicles += unauth_increment
            total_vehicles += unauth_increment
            # Trigger Streamlit Alert
            alert_placeholder.error("🚨 ALERT: Unauthorized Vehicle Detected crossing the toll line!")
        else:
            alert_placeholder.empty()

        # Update Dashboard Metrics
        metric_total.metric("Total Traffic", total_vehicles)
        metric_auth.metric("Authorized (Paid)", auth_vehicles)
        metric_unauth.metric("Unauthorized (Alert)", unauth_vehicles, delta_color="inverse")
        metric_rev.metric("Toll Revenue (BDT)", f"৳{total_revenue}")

        # Display Video - smaller size
        col_video, col_empty = st.columns([3, 1])
        with col_video:
            video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()

# --- 4. Standard Predict (For Images Only - No Tracking) ---
def predict_image(model, image, conf_thresh):
    results = model.predict(image, conf=conf_thresh)
    annotated_img = image.copy()
    counts = {}
    
    if results:
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        
        annotated_img = r.plot() # Use built-in plotter for simple images
        
        for cls in classes:
            name = model.names[int(cls)]
            counts[name] = counts.get(name, 0) + 1
            
    return annotated_img, counts

# --- 5. Main App Layout ---
def main():
    st.title("🚦 Smart Vehicle Detection and Toll Management")
    st.write("Detect vehicles, collect toll, and identify unauthorized entries.")

    # --- Sidebar: Configuration ---
    st.sidebar.header("1. Model & Detection")
    selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_PATHS.keys()))
    model_path = MODEL_PATHS[selected_model_name]
    
    # Load Model
    if os.path.exists(model_path) or "yolov8n" in model_path:
        model, err = load_model(model_path)
        if err:
            st.error(f"Error loading model: {err}")
            st.stop()
    else:
        st.error("Model file not found.")
        st.stop()

    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45)
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. Toll Configuration")
    
    # Toll Settings
    class_names = list(model.names.values())
    default_unauthorized = [name for name in class_names if name.lower() in UNAUTHORIZED_DEFAULTS]
    unauthorized_classes = st.sidebar.multiselect(
        "Select Unauthorized Classes (Alert)",
        class_names,
        default=default_unauthorized,
    )

    unauthorized_lookup = {name.lower() for name in unauthorized_classes}

    st.sidebar.subheader("Toll Fees per Vehicle (BDT)")
    fallback_toll = st.sidebar.number_input(
        "Fallback Toll (other vehicles)",
        min_value=0,
        value=DEFAULT_TOLL_RATES.get("_default", 0),
        step=10,
    )

    toll_rates = {"_default": fallback_toll}
    for name in class_names:
        key = name.lower()
        default_value = DEFAULT_TOLL_RATES.get(key, fallback_toll)
        label = name.replace("_", " ").title()
        toll_rates[key] = st.sidebar.number_input(
            f"{label} Toll",
            min_value=0,
            value=default_value,
            step=10,
        )

    line_position = st.sidebar.slider("Line Position (Screen Height %)", 0.1, 0.9, 0.65)

    # --- Main Area ---
    source_type = st.radio("Select Input Source:", ["Video", "Webcam", "Image"], horizontal=True)

    if source_type == "Image":
        st.info("Note: Toll counting and tracking features are disabled for static images.")
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if img_file:
            image = Image.open(img_file).convert('RGB')
            img_array = np.array(image)
            annotated_img, counts = predict_image(model, img_array, conf_threshold)
            st.image(annotated_img, caption="Result", use_container_width=True)
            st.json(counts)

    elif source_type == "Video":
        vid_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if vid_file:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(vid_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            if st.button("▶ Start Toll System"):
                process_video_stream(
                    model,
                    cap,
                    conf_threshold,
                    line_position,
                    unauthorized_lookup,
                    toll_rates,
                )

    elif source_type == "Webcam":
        if st.button("▶ Start Webcam System"):
            cap = cv2.VideoCapture(0) # 0 is usually default webcam
            process_video_stream(
                model,
                cap,
                conf_threshold,
                line_position,
                unauthorized_lookup,
                toll_rates,
            )

if __name__ == "__main__":
    main()