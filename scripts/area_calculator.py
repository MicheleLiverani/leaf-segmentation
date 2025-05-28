"""
Area Measurement Tool

A production-ready tool for measuring areas in images by selecting reference points
and calculating pixel-to-cm conversion. Supports batch processing and Excel output.

Usage:
    python area_calculator.py --data-path /path/to/data --points-file points.json
    python area_calculator.py -d /path/to/data -p points.json --padding 1000 --reference-cm 1.0
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Set matplotlib backend early
matplotlib.use('TkAgg')


class AreaCalculator:
    """Interactive area measurement tool with reference point selection."""
    
    def __init__(self, reference_cm: float = 1.0, figure_size: Tuple[int, int] = (14, 14)):
        self.reference_cm = reference_cm
        self.figure_size = figure_size
        self.points = []
        self.current_image = None
        self.fig = None
        self.ax = None
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from path."""
        try:
            img = Image.open(image_path)
            return np.array(img)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    def _onclick(self, event):
        """Handle mouse click events for reference point selection."""
        x, y = event.xdata, event.ydata
        
        if x is None or y is None:
            return
            
        if event.button == 3 and len(self.points) < 2:  # Right click only
            point = {'x': float(x), 'y': float(y)}
            self.points.append(point)
            
            logging.info(f"Reference point added: ({x:.2f}, {y:.2f})")
            
            # Plot the point
            self.ax.plot(x, y, '+', color='black', markersize=10, markeredgewidth=2)
            plt.draw()
            
            # Disable further clicks if two points are selected
            if len(self.points) == 2:
                self.fig.canvas.mpl_disconnect(self.cid_click)
                logging.info("Two reference points selected. Press Enter to continue or Z to undo.")
    
    def _onkey(self, event):
        """Handle keyboard events."""
        if event.key == 'z' and self.points:
            # Undo last point
            removed_point = self.points.pop()
            logging.info(f"Reference point removed: ({removed_point['x']:.2f}, {removed_point['y']:.2f})")
            
            # Redraw image and remaining points
            self._redraw_points()
            
            # Re-enable clicking if less than 2 points
            if len(self.points) < 2:
                self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
                
        elif event.key == 'enter':
            # Finish selection
            if len(self.points) == 2:
                plt.close(self.fig)
            else:
                logging.warning("Need exactly 2 reference points. Current points: {}".format(len(self.points)))
        elif event.key == 'escape':
            # Cancel selection
            self.points.clear()
            plt.close(self.fig)
    
    def _redraw_points(self):
        """Redraw the image and all points."""
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.set_title(f"{Path(self.current_image_name).stem} - Right click: Add reference point ({self.reference_cm}cm), Z: Undo, Enter: Continue")
        
        for point in self.points:
            self.ax.plot(point['x'], point['y'], '+', color='black', markersize=10, markeredgewidth=2)
        
        plt.draw()
    
    def select_reference_points(self, image_path: Path) -> Optional[List[Dict]]:
        """Select reference points for pixel-to-cm conversion."""
        try:
            self.current_image = self._load_image(image_path)
            self.current_image_name = image_path.name
            self.points = []
            
            h, w = self.current_image.shape[:2]
            ratio = h / w
            
            # Create figure
            self.fig, self.ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[0] * ratio))
            
            # Make full screen if possible
            try:
                self.fig.canvas.manager.full_screen_toggle()
            except AttributeError:
                pass  # Full screen not available on all backends
                
            self.ax.imshow(self.current_image)
            self.ax.set_title(f"{image_path.stem} - Right click: Add reference point ({self.reference_cm}cm), Z: Undo, Enter: Continue")
            
            # Connect event handlers
            self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
            self.fig.canvas.mpl_connect('key_press_event', self._onkey)
            
            plt.show()
            
            if len(self.points) == 2:
                return self.points
            return None
            
        except Exception as e:
            logging.error(f"Error selecting reference points for {image_path}: {e}")
            return None
        finally:
            if self.fig:
                plt.close(self.fig)
    
    def calculate_area(self, image_path: Path, reference_points: List[Dict]) -> float:
        """Calculate area in cm² using reference points and red pixel mask."""
        try:
            # Load image
            img = self._load_image(image_path)
            
            # Calculate pixels per cm using reference points
            distance_pixels = np.linalg.norm([
                reference_points[0]["x"] - reference_points[1]["x"],
                reference_points[0]["y"] - reference_points[1]["y"]
            ])
            cms_per_pixel = self.reference_cm / distance_pixels
            
            # Create red pixel mask
            mask_red = np.logical_and.reduce([
                img[..., 0] == 255,  # Red channel = 255
                img[..., 1] == 0,    # Green channel = 0
                img[..., 2] == 0     # Blue channel = 0
            ]).astype(np.uint8)
            
            # Calculate area
            red_pixel_count = mask_red.sum()
            area_cm2 = red_pixel_count * (cms_per_pixel ** 2)
            
            logging.info(f"Red pixels: {red_pixel_count}, Area: {area_cm2:.4f} cm²")
            return area_cm2
            
        except Exception as e:
            logging.error(f"Error calculating area for {image_path}: {e}")
            raise


class AreaMeasurementManager:
    """Manages the area measurement workflow and data persistence."""
    
    def __init__(self, data_path: Path, output_excel: Optional[Path] = None,
                 reference_cm: float = 1.0, date_pattern: str = None):
        self.data_path = Path(data_path)
        self.output_excel = Path(output_excel) if output_excel else self.data_path / "area_measurements.xlsx"
        self.date_pattern = date_pattern or r"leaf expansion (\d{2})-(\d{2})_(\d+)(am|pm)"
        
        self.calculator = AreaCalculator(reference_cm=reference_cm)
        
        # Define paths
        self.red_paint_path = self.data_path / "red_paint"
        self.reference_points_file = self.red_paint_path / "reference_points.json"
        
        # Validate paths
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        if not self.red_paint_path.exists():
            raise ValueError(f"Red paint directory does not exist: {self.red_paint_path}")
        
        # Load or initialize reference points
        self.reference_points = self._load_json(self.reference_points_file) if self.reference_points_file.exists() else {}
        
        # Load or initialize Excel dataframe
        self.df = self._load_or_create_excel()
    
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Could not load JSON file {file_path}: {e}")
            return {}
    
    def _save_json(self, data: Dict, file_path: Path):
        """Save JSON data to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Data saved to {file_path}")
        except Exception as e:
            logging.error(f"Could not save JSON file {file_path}: {e}")
    
    def _load_or_create_excel(self) -> pd.DataFrame:
        """Load existing Excel file or create new dataframe."""
        if self.output_excel.exists():
            try:
                return pd.read_excel(self.output_excel)
            except Exception as e:
                logging.warning(f"Could not load existing Excel file: {e}")
        
        # Create new dataframe with required columns
        return pd.DataFrame(columns=[
            "exp_name", "date", "trolley", "treatment", "#plant", 
            "area [cm^2]", "raw_img_path", "red_img_path"
        ])
    
    def _save_excel(self):
        """Save dataframe to Excel file."""
        try:
            self.output_excel.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_excel(self.output_excel, index=False)
            logging.info(f"Excel data saved to {self.output_excel}")
        except Exception as e:
            logging.error(f"Could not save Excel file: {e}")
    
    def _find_red_images(self) -> List[Path]:
        """Find all red paint images to process."""
        images = []
        for ext in ['png', 'jpg', 'jpeg']:
            images.extend(self.red_paint_path.rglob(f"*.{ext}"))
        return sorted(images)
    
    def _extract_metadata(self, image_path: Path) -> Dict:
        """Extract metadata from image path and experiment name."""
        exp_name = image_path.parent.name
        
        # Find corresponding raw image
        raw_img_path = None
        raw_search_path = self.data_path / "raw" / exp_name
        if raw_search_path.exists():
            raw_candidates = list(raw_search_path.glob(f"{image_path.stem}.*"))
            if raw_candidates:
                raw_img_path = raw_candidates[0]
        
        # Extract date information
        date = None
        try:
            groups = re.findall(self.date_pattern, exp_name)
            if groups:
                day, month, hour, period = groups[0]
                date_str = f"{day}-{month}-24 {hour}{period}"
                date = pd.to_datetime(date_str, format="%d-%m-%y %I%p")
        except Exception as e:
            logging.warning(f"Could not parse date from {exp_name}: {e}")
        
        return {
            "exp_name": exp_name,
            "raw_img_path": str(raw_img_path) if raw_img_path else None,
            "red_img_path": str(image_path),
            "date": date,
            "trolley": None,  # Placeholder
            "treatment": None,  # Placeholder
            "#plant": None  # Placeholder
        }
    
    def _is_already_processed(self, image_path: Path) -> bool:
        """Check if image is already processed in Excel file."""
        return str(image_path) in self.df["red_img_path"].tolist()
    
    def run_measurement_session(self):
        """Run the main area measurement session."""
        images = self._find_red_images()
        
        if not images:
            logging.warning(f"No images found in {self.red_paint_path}")
            return
        
        logging.info(f"Found {len(images)} images to process")
        
        processed_count = 0
        skipped_count = 0
        
        for i, image_path in enumerate(images, 1):
            image_name = image_path.stem
            
            # Check if already processed
            if self._is_already_processed(image_path):
                logging.info(f"[{i}/{len(images)}] Skipping {image_name} (already processed)")
                skipped_count += 1
                continue
            
            logging.info(f"[{i}/{len(images)}] Processing {image_name}")
            
            try:
                # Get or select reference points
                if image_name in self.reference_points:
                    ref_points = self.reference_points[image_name]
                    logging.info(f"Using existing reference points for {image_name}")
                else:
                    logging.info(f"Selecting reference points for {image_name}")
                    ref_points = self.calculator.select_reference_points(image_path)
                    
                    if not ref_points:
                        logging.warning(f"No reference points selected for {image_name}")
                        continue
                    
                    # Save reference points
                    self.reference_points[image_name] = ref_points
                    self._save_json(self.reference_points, self.reference_points_file)
                
                # Calculate area
                area = self.calculator.calculate_area(image_path, ref_points)
                
                # Extract metadata
                metadata = self._extract_metadata(image_path)
                metadata["area [cm^2]"] = area
                
                # Add to dataframe
                self.df = pd.concat([self.df, pd.DataFrame([metadata])], axis=0, ignore_index=True)
                
                # Save Excel file
                self._save_excel()
                
                processed_count += 1
                logging.info(f"Successfully processed {image_name}: {area:.4f} cm²")
                
            except KeyboardInterrupt:
                logging.info("Measurement session interrupted by user")
                break
            except Exception as e:
                logging.error(f"Error processing {image_name}: {e}")
                continue
        
        logging.info(f"Session complete. Processed: {processed_count}, Skipped: {skipped_count}")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive area measurement tool for red-painted regions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --data-path ./data/replicate_3
  python %(prog)s -d ./data --reference-cm 2.0
  python %(prog)s -d ./data --output-excel measurements.xlsx --verbose

Controls:
  Right click: Add reference point (exactly 2 needed)
  Z key:       Undo last reference point
  Enter:       Continue with selected points
  Escape:      Cancel current image

The tool expects:
  - A 'red_paint' subdirectory with images to measure
  - Optional 'raw' subdirectory with original images
        """
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to data directory containing red_paint subdirectory'
    )
    
    parser.add_argument(
        '--output-excel', '-o',
        type=str,
        help='Path to output Excel file (default: data_path/area_measurements.xlsx)'
    )
    
    parser.add_argument(
        '--reference-cm',
        type=float,
        default=1.0,
        help='Distance in cm between reference points (default: 1.0)'
    )
    
    parser.add_argument(
        '--date-pattern',
        type=str,
        help='Regex pattern for extracting date from experiment names'
    )
    
    parser.add_argument(
        '--figure-size',
        nargs=2,
        type=int,
        default=[14, 14],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for measurement window (default: 14 14)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Create measurement manager and run session
        manager = AreaMeasurementManager(
            data_path=args.data_path,
            output_excel=args.output_excel,
            reference_cm=args.reference_cm,
            date_pattern=args.date_pattern
        )
        
        # Update figure size in calculator
        manager.calculator.figure_size = tuple(args.figure_size)
        
        logging.info("Starting area measurement session...")
        manager.run_measurement_session()
        
    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()