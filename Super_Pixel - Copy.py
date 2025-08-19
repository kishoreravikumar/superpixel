import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
from scipy.spatial.distance import cdist
from collections import defaultdict
import imageio
import os

class FelzenszwalbSuperpixel:
    def __init__(self, image, scale=100, sigma=0.5, min_size=50):
        """
        Initialize Felzenszwalb superpixel segmentation
        
        Args:
            image: Input image (grayscale or RGB)
            scale: Scale parameter for segmentation
            sigma: Gaussian smoothing parameter
            min_size: Minimum component size
        """
        self.original_image = image.copy()
        if len(image.shape) == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            self.image = image.copy()
        
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.height, self.width = self.image.shape
        
        # For animation
        self.frames = []
        self.current_segments = None
        
    def gaussian_blur(self, image, sigma):
        """Apply Gaussian blur to the image"""
        if sigma > 0:
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return image
    
    def create_graph(self, image):
        """Create a graph from the image with edge weights based on intensity differences"""
        edges = []
        
        # Create edges for 4-connectivity (up, down, left, right)
        for y in range(self.height):
            for x in range(self.width):
                current_pixel = image[y, x]
                
                # Right neighbor
                if x < self.width - 1:
                    neighbor_pixel = image[y, x + 1]
                    weight = abs(float(current_pixel) - float(neighbor_pixel))
                    edges.append((y * self.width + x, y * self.width + (x + 1), weight))
                
                # Bottom neighbor
                if y < self.height - 1:
                    neighbor_pixel = image[y + 1, x]
                    weight = abs(float(current_pixel) - float(neighbor_pixel))
                    edges.append((y * self.width + x, (y + 1) * self.width + x, weight))
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        return edges
    
    def find_parent(self, parent, i):
        """Find the root of the set containing element i (with path compression)"""
        if parent[i] != i:
            parent[i] = self.find_parent(parent, parent[i])
        return parent[i]
    
    def union_sets(self, parent, rank, threshold, x, y, weight):
        """Union two sets if the edge weight is less than the threshold"""
        root_x = self.find_parent(parent, x)
        root_y = self.find_parent(parent, y)
        
        if root_x != root_y and weight < min(threshold[root_x], threshold[root_y]):
            # Union by rank
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
                threshold[root_y] = weight + self.scale / rank[root_y]
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
                threshold[root_x] = weight + self.scale / rank[root_x]
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
                threshold[root_x] = weight + self.scale / rank[root_x]
            return True
        return False
    
    def segment_image_with_animation(self):
        """Perform segmentation and capture frames for animation"""
        # Apply Gaussian blur
        blurred_image = self.gaussian_blur(self.image, self.sigma)
        
        # Create graph
        edges = self.create_graph(blurred_image)
        
        # Initialize Union-Find structure
        num_pixels = self.height * self.width
        parent = list(range(num_pixels))
        rank = [1] * num_pixels
        threshold = [self.scale] * num_pixels
        
        # Initialize segments
        segments = np.arange(num_pixels).reshape(self.height, self.width)
        
        # Save initial frame
        self.save_frame(segments, 0)
        
        frame_count = 1
        edges_processed = 0
        
        # Process edges
        for edge_idx, (u, v, weight) in enumerate(edges):
            merged = self.union_sets(parent, rank, threshold, u, v, weight)
            edges_processed += 1
            
            # Update segments periodically for animation
            if edges_processed % max(1, len(edges) // 50) == 0 or merged:
                # Update segment labels
                for y in range(self.height):
                    for x in range(self.width):
                        pixel_id = y * self.width + x
                        segments[y, x] = self.find_parent(parent, pixel_id)
                
                # Save frame every 10 edges for faster GIF creation
                if edge_idx % 10 == 0:
                    self.save_frame(segments, frame_count)
                    frame_count += 1
        
        # Final cleanup - merge small components
        final_segments = self.merge_small_components(segments, parent)
        self.save_frame(final_segments, frame_count)
        
        self.current_segments = final_segments
        return final_segments
    
    def merge_small_components(self, segments, parent):
        """Merge components smaller than min_size"""
        # Count component sizes
        component_sizes = defaultdict(int)
        for y in range(self.height):
            for x in range(self.width):
                root = self.find_parent(parent, segments[y, x])
                component_sizes[root] += 1
        
        # Merge small components with neighbors
        final_segments = segments.copy()
        for y in range(self.height):
            for x in range(self.width):
                root = self.find_parent(parent, segments[y, x])
                if component_sizes[root] < self.min_size:
                    # Find the largest neighboring component
                    neighbors = []
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            neighbor_root = self.find_parent(parent, segments[ny, nx])
                            if neighbor_root != root:
                                neighbors.append(neighbor_root)
                    
                    if neighbors:
                        # Choose the most frequent neighbor
                        best_neighbor = max(set(neighbors), key=neighbors.count)
                        final_segments[y, x] = best_neighbor
        
        return final_segments
    
    def save_frame(self, segments, frame_num):
        """Save current segmentation state as a frame"""
        # Create colored visualization
        colored_segments = self.colorize_segments(segments)
        self.frames.append(colored_segments.copy())
    
    def colorize_segments(self, segments):
        """Create a colored visualization of segments"""
        # Get unique segment IDs
        unique_segments = np.unique(segments)
        
        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_segments)))
        
        # Create colored image
        colored = np.zeros((self.height, self.width, 3))
        for i, seg_id in enumerate(unique_segments):
            mask = segments == seg_id
            colored[mask] = colors[i][:3]
        
        return colored
    
    def create_gif(self, output_path="superpixel_animation.gif", duration=0.2):
        """Create GIF animation from saved frames"""
        if not self.frames:
            print("No frames to create animation. Run segmentation first.")
            return
        
        # Convert frames to uint8
        gif_frames = []
        for frame in self.frames:
            gif_frame = (frame * 255).astype(np.uint8)
            gif_frames.append(gif_frame)
        
        # Save as GIF
        imageio.mimsave(output_path, gif_frames, duration=duration)
        print(f"Animation saved as {output_path}")
    
    def visualize_final_result(self):
        """Visualize the final segmentation result"""
        if self.current_segments is None:
            print("No segmentation result. Run segmentation first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        if len(self.original_image.shape) == 3:
            axes[0].imshow(self.original_image)
        else:
            axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmented image
        colored_segments = self.colorize_segments(self.current_segments)
        axes[1].imshow(colored_segments)
        axes[1].set_title(f'Superpixel Segmentation\n({len(np.unique(self.current_segments))} segments)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('superpixel_result.png', dpi=150, bbox_inches='tight')
        plt.show()

def create_sample_image():
    """Create a sample image for demonstration"""
    # Create a simple synthetic image with different intensity regions
    image = np.zeros((100, 100), dtype=np.uint8)
    
    # Add some regions with different intensities
    image[20:40, 20:40] = 100  # Dark gray square
    image[60:80, 60:80] = 200  # Light gray square
    image[20:40, 60:80] = 150  # Medium gray square
    image[60:80, 20:40] = 50   # Very dark gray square
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image

def load_image_from_file(file_path):
    """Load image from file"""
    try:
        image = cv2.imread(file_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            print(f"Could not load image from {file_path}")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    """Main function to run the superpixel segmentation with animation"""
    print("Felzenszwalb Superpixel Segmentation with Animation")
    print("=" * 50)
    
    # Option 1: Use a sample image
    print("Creating sample image...")
    image = create_sample_image()
    
    # Option 2: Uncomment to use your own image
    # image_path = "your_image.jpg"  # Replace with your image path
    # image = load_image_from_file(image_path)
    # if image is None:
    #     print("Using sample image instead...")
    #     image = create_sample_image()
    
    # Create superpixel segmentation object
    segmenter = FelzenszwalbSuperpixel(
        image=image,
        scale=50,      # Lower scale = more segments
        sigma=0.5,     # Gaussian smoothing
        min_size=10    # Minimum segment size
    )
    
    print("Running segmentation with animation...")
    segments = segmenter.segment_image_with_animation()
    
    print(f"Segmentation complete! Found {len(np.unique(segments))} segments")
    
    # Create GIF animation
    print("Creating GIF animation...")
    segmenter.create_gif("felzenszwalb_superpixel_animation.gif", duration=0.3)
    
    # Visualize final result
    print("Displaying final result...")
    segmenter.visualize_final_result()
    
    print("\nFiles created:")
    print("- felzenszwalb_superpixel_animation.gif (animation)")
    print("- superpixel_result.png (final result)")

if __name__ == "__main__":
    main()
