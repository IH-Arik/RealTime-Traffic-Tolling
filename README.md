# Bangladesh Smart Toll & Traffic System

A computer vision-based traffic monitoring and toll collection system using YOLO object detection and Streamlit for the web interface.

## Features

- Real-time vehicle detection and tracking
- Automatic toll collection based on vehicle type
- Unauthorized vehicle alerts (rickshaw, CNG, van)
- Per-vehicle toll rate configuration
- Support for video, webcam, and image inputs
- Live traffic statistics and revenue tracking

## Deployment on Hugging Face Spaces

### Prerequisites

- Hugging Face account
- Model files (best.pt, best150.pt) uploaded to your Space

### Quick Deploy

1. **Create a new Space** on Hugging Face
2. **Choose Docker** as the SDK
3. **Upload these files** to your Space:
   - `app.py`
   - `Dockerfile`
   - `requirements.txt`
   - `.dockerignore`
   - Model files (`*.pt`)

### Docker Configuration

The `Dockerfile` is optimized for Hugging Face Spaces:
- Uses Python 3.9 slim base image
- Installs OpenCV and system dependencies
- Configures Streamlit for headless operation
- Exposes port 8501 (Hugging Face standard)
- Includes health checks

### Model Files

Upload your YOLO model files to the Space root directory:
- `best.pt` - Main trained model
- `best150.pt` - Alternative model
- The app will fallback to YOLOv8n if models are missing

### Environment Variables (Optional)

You can set these in your Space settings:
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Usage

1. **Select Model** - Choose your trained YOLO model
2. **Configure Toll** - Set unauthorized vehicles and toll rates
3. **Choose Input** - Video file, webcam, or static image
4. **Start Monitoring** - Real-time detection and toll collection

## System Requirements

- Python 3.9+
- OpenCV support
- GPU acceleration (optional but recommended)
- Minimum 4GB RAM

## License

MIT License - see LICENSE file for details
