"""
Image Point Annotation Tool

A production-ready tool for annotating points on images with left/right click labels.
Supports batch processing and resumable annotation sessions.

Usage:
    python image_annotator.py --data-path /path/to/images --output-path /path/to/results
    python image_annotator.py -d /path/to/images -o /path/to/results --extensions jpg png heic
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Set matplotlib backend early
matplotlib.use('TkAgg')

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False
    logging.warning("pillow_heif not available. HEIC/HEIF support disabled.")


class ImageAnnotator:
    """Interactive image point annotation tool."""
    
    def __init__(self, figure_size: Tuple[int, int] = (14, 14)):
        self.figure_size = figure_size
        self.points = []
        self.current_image = None
        self.fig = None
        self.ax = None
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from path, handling different formats."""
        try:
            if image_path.suffix.lower() == ".heic":
                if not HEIF_SUPPORT:
                    raise ValueError(f"HEIC support not available for {image_path}")
                img_pil = Image.open(image_path)
                img = np.array(img_pil)
                # Convert RGBA to RGB if necessary
                if img.shape[2] == 4:
                    img = img[:, :, :3]
            else:
                img = cv2.imread(str(image_path))
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
    
    def _onclick(self, event):
        """Handle mouse click events."""
        x, y = event.xdata, event.ydata
        
        if x is None or y is None:
            return
            
        if event.button == 1:  # Left click - positive label
            label = 1
            color = 'g'
        elif event.button == 3:  # Right click - negative label
            label = 0
            color = 'r'
        else:
            return
            
        point = {'x': float(x), 'y': float(y), 'label': label}
        self.points.append(point)
        
        logging.info(f"Point added: ({x:.2f}, {y:.2f}), Label: {label}")
        
        # Plot the point
        self.ax.plot(x, y, 'o', color=color, markersize=8)
        plt.draw()
    
    def _onkey(self, event):
        """Handle keyboard events."""
        if event.key == 'z' and self.points:
            # Undo last point
            removed_point = self.points.pop()
            logging.info(f"Point removed: ({removed_point['x']:.2f}, {removed_point['y']:.2f})")
            
            # Redraw image and remaining points
            self._redraw_points()
        elif event.key == 'enter':
            # Finish annotation
            plt.close(self.fig)
        elif event.key == 'escape':
            # Cancel annotation
            self.points.clear()
            plt.close(self.fig)
    
    def _redraw_points(self):
        """Redraw the image and all points."""
        self.ax.clear()
        self.ax.imshow(self.current_image)
        self.ax.set_title(f"{Path(self.current_image_name).stem} - Left: Positive, Right: Negative, Z: Undo, Enter: Save, Esc: Cancel")
        
        for point in self.points:
            color = 'g' if point['label'] == 1 else 'r'
            self.ax.plot(point['x'], point['y'], 'o', color=color, markersize=8)
        
        plt.draw()
    
    def annotate_image(self, image_path: Path) -> Optional[Dict]:
        """Annotate a single image and return the points."""
        try:
            self.current_image = self._load_image(image_path)
            self.current_image_name = image_path.name
            self.points = []
            
            h, w = self.current_image.shape[:2]
            ratio = h / w
            
            # Create figure
            self.fig, self.ax = plt.subplots(figsize=(self.figure_size[0], self.figure_size[0] * ratio))
            self.ax.imshow(self.current_image)
            self.ax.set_title(f"{image_path.stem} - Left: Positive, Right: Negative, Z: Undo, Enter: Save, Esc: Cancel")
            
            # Connect event handlers
            self.fig.canvas.mpl_connect('button_press_event', self._onclick)
            self.fig.canvas.mpl_connect('key_press_event', self._onkey)
            
            plt.show()
            
            if self.points:
                return {image_path.stem: self.points}
            return None
            
        except Exception as e:
            logging.error(f"Error annotating image {image_path}: {e}")
            return None
        finally:
            if self.fig:
                plt.close(self.fig)


class AnnotationManager:
    """Manages the annotation workflow and data persistence."""
    
    def __init__(self, data_path: Path, output_path: Path, extensions: List[str]):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.extensions = [ext.lower().lstrip('.') for ext in extensions]
        self.points_file = self.output_path / "points.json"
        self.annotator = ImageAnnotator()
        
        # Validate paths
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing annotations
        self.annotations = self._load_annotations()
    
    def _load_annotations(self) -> Dict:
        """Load existing annotations from file."""
        if self.points_file.exists():
            try:
                with open(self.points_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load existing annotations: {e}")
        return {}
    
    def _save_annotations(self):
        """Save annotations to file."""
        try:
            with open(self.points_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            logging.info(f"Annotations saved to {self.points_file}")
        except Exception as e:
            logging.error(f"Could not save annotations: {e}")
    
    def _find_images(self) -> List[Path]:
        """Find all images matching the specified extensions."""
        images = []
        for ext in self.extensions:
            pattern = f"*.{ext}"
            images.extend(self.data_path.rglob(pattern))
        
        # Sort for consistent ordering
        return sorted(images)
    
    def _copy_image(self, image_path: Path):
        """Copy image to processed directory, converting HEIC to JPG if needed."""
        try:
            exp_name = image_path.parent.name
            exp_processed_path = self.output_path / exp_name
            exp_processed_path.mkdir(parents=True, exist_ok=True)
            
            if image_path.suffix.lower() == ".heic":
                # Convert HEIC to JPG
                if HEIF_SUPPORT:
                    img_pil = Image.open(image_path)
                    output_path = exp_processed_path / f"{image_path.stem}.jpg"
                    img_pil.save(output_path, "JPEG", quality=90)
                    logging.info(f"Converted HEIC to JPG: {output_path}")
                else:
                    logging.warning(f"Skipping HEIC conversion for {image_path} (no HEIF support)")
            else:
                # Copy original file
                output_path = exp_processed_path / image_path.name
                shutil.copy2(image_path, output_path)
                logging.info(f"Copied image: {output_path}")
                
        except Exception as e:
            logging.error(f"Could not copy image {image_path}: {e}")
    
    def run_annotation_session(self):
        """Run the main annotation session."""
        images = self._find_images()
        
        if not images:
            logging.warning(f"No images found with extensions {self.extensions} in {self.data_path}")
            return
        
        logging.info(f"Found {len(images)} images to process")
        
        processed_count = 0
        skipped_count = 0
        
        for i, image_path in enumerate(images, 1):
            image_name = image_path.stem
            
            # Skip already annotated images
            if image_name in self.annotations:
                logging.info(f"[{i}/{len(images)}] Skipping {image_name} (already annotated)")
                skipped_count += 1
                continue
            
            logging.info(f"[{i}/{len(images)}] Processing {image_name}")
            
            try:
                # Annotate the image
                result = self.annotator.annotate_image(image_path)
                
                if result:
                    # Save annotations
                    self.annotations.update(result)
                    self._save_annotations()
                    
                    # Copy/convert image
                    self._copy_image(image_path)
                    
                    processed_count += 1
                    logging.info(f"Successfully processed {image_name}")
                else:
                    logging.info(f"No points annotated for {image_name}")
                    
            except KeyboardInterrupt:
                logging.info("Annotation session interrupted by user")
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
        description="Interactive image point annotation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run %(prog)s --data-path ./images --output-path ./results
  uv run %(prog)s -d ./images -o ./results --extensions jpg png heic
  uv run %(prog)s -d ./images -o ./results --figure-size 12 10 --verbose

Controls:
  Left click:  Add positive point (green)
  Right click: Add negative point (red)
  Z key:       Undo last point
  Enter:       Save and continue to next image
  Escape:      Cancel current image
        """
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to directory containing images'
    )
    
    parser.add_argument(
        '--output-path', '-o',
        type=str,
        required=True,
        help='Path to output directory for results'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['jpg', 'jpeg', 'png', 'heic'],
        help='Image file extensions to process (default: jpg jpeg png heic)'
    )
    
    parser.add_argument(
        '--figure-size',
        nargs=2,
        type=int,
        default=[14, 14],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size for annotation window (default: 14 14)'
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
        # Create annotation manager and run session
        manager = AnnotationManager(
            data_path=args.data_path,
            output_path=args.output_path,
            extensions=args.extensions
        )
        
        # Update figure size in annotator
        manager.annotator.figure_size = tuple(args.figure_size)
        
        logging.info("Starting annotation session...")
        manager.run_annotation_session()
        
    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()