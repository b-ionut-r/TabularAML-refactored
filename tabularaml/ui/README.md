# TabularAML Feature Generator UI

Beautiful web-based interface for the TabularAML FeatureGenerator with real-time progress tracking.

## Features

🎛️ **Dual Mode Interface:**
- **Train Mode**: Full parameter control with real-time progress
- **Transform Mode**: Load saved generators and transform new data

🎨 **Beautiful Interface:**
- Modern responsive design
- Real-time progress bars and status cards
- Parameter tooltips for guidance
- Live log output and feature tracking

⚙️ **Comprehensive Controls:**
- All FeatureGenerator parameters exposed
- Generation modes: Low, Medium, Best, Extreme, None (Custom)
- Advanced parameters with tooltips
- Save/Load functionality

## Quick Start

```bash
# Install dependencies
pip install Flask Flask-SocketIO pandas numpy

# Run the server
cd tabularaml/ui
python feature_generator_server.py

# Open browser
# http://localhost:5000
```

## Files

- `feature_generator_ui.html` - Complete web interface
- `feature_generator_server.py` - Flask server with real progress tracking
- `README.md` - This documentation

## Usage

### Train Mode
1. Upload your dataset (CSV/JSON)
2. Select target column
3. Configure parameters (or use presets)
4. Click "Start Feature Generation"
5. Watch real-time progress
6. Save trained generator

### Transform Mode
1. Load a saved generator (.pkl file)
2. Upload dataset to transform
3. Click "Transform Data"
4. Download transformed results

## Technical Details

- **Real Progress Tracking**: Intercepts actual FeatureGenerator logs and progress
- **WebSocket Communication**: Live updates via Socket.IO
- **Comprehensive Parameters**: All FeatureGenerator init parameters exposed
- **Save/Load Support**: Persistent generator storage with cloudpickle