# Leaf Area Measurement with SAM

A complete pipeline for measuring leaf areas using Meta's Segment Anything Model (SAM). This tool provides an interactive workflow to annotate control points, segment leaves using SAM, and calculate precise area measurements.

## 🌿 Overview

This project implements a three-stage pipeline:

1. **Point Annotation**: Interactive GUI for marking positive/negative control points on leaf images
2. **Leaf Segmentation**: Uses Meta's SAM model to segment leaves based on control points
3. **Area Calculation**: Calculates leaf areas in cm² using reference measurements

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for SAM)
- 8GB+ RAM
- Operating System: Windows, macOS, or Linux

### Installation

```bash
# Clone the repository
git https://github.com/MicheleLiverani/leaf-segmentation.git
cd leaf-segmentation

# Create virtual environment and install dependencies
uv sync
```

## 📋 Usage

### 1. Annotate your images

Provide control points to guide SAM to an accurate segmentation.

```bash
# Run the image annotator script
uv run scripts/image_annotator.py -d DATA_PATH -o OUTPUT_PATH
```

**Controls:**
- **Left click**: Add positive point (green) - inside leaf area
- **Right click**: Add negative point (red) - outside leaf area
- **Z key**: Undo last point
- **Enter**: Save and continue to next image
- **Escape**: Cancel current image

### 2. Predict the images

Run `notebooks/leaf_segmentation.ipynb` on Colab and select a GPU backend.

#### Step 3: Calculate Areas

Copy the segmented images by the notebook locally and run the area calculator script.

```bash
# Measure leaf areas
uv run scripts/area_calculator.py -d DATA_PATH -o OUTPUT.xlsx
```

**Controls:**
- **Right click**: Add reference point (need exactly 2 points)
- **Z key**: Undo last reference point
- **Enter**: Continue with selected reference points
- **Escape**: Cancel current image

## 📁 Data Organization

### Input Structure
```
data/raw/
├── experiment_1/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── experiment_2/
    ├── image001.jpg
    └── ...
```

### Output Structure
```
data/
├── processed/
│   ├── experiment_1/
│   │   ├── image001.jpg
│   │   └── ...
│   └── points.json              # Annotation points
├── segmented/
│   └── red_paint/
│       ├── experiment_1/
│       │   ├── image001.png     # Red-masked images
│       │   └── ...
│       └── reference_points.json
└── results/
    ├── area_measurements.xlsx   # Final measurements
    └── summary_report.html
```

## 📊 Results

### Excel Output

The final measurements are saved in Excel format with columns:
- `exp_name`: Experiment name
- `date`: Extracted date/time
- `trolley`: Equipment identifier
- `treatment`: Experimental treatment
- `#plant`: Plant number
- `area [cm^2]`: Calculated leaf area
- `raw_img_path`: Original image path
- `red_img_path`: Segmented image path

### Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_excel('data/results/area_measurements.xlsx')

# Plot area over time
df.groupby('date')['area [cm^2]'].mean().plot()
plt.title('Average Leaf Area Over Time')
plt.ylabel('Area (cm²)')
plt.show()
```

#### HEIC Image Support
```bash
# Install HEIC support
uv add pillow-heif
```

#### Matplotlib Backend Issues
```bash
# On headless servers
export MPLBACKEND=Agg

# Or install GUI backend
sudo apt-get install python3-tk  # Ubuntu/Debian
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [OpenCV](https://opencv.org/) for image processing
- [Matplotlib](https://matplotlib.org/) for visualization
- [Pandas](https://pandas.pydata.org/) for data handling

---

**Happy leaf measuring! 🌱📏**