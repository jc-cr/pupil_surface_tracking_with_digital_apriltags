import cv2
import numpy as np
import cairosvg
import io
from PIL import Image

class Video:
    def __init__(self):
        pass

    def get_frame(self):
        raise NotImplementedError("Subclass must implement abstract method")

class EmptyVideo(Video):
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

    def get_frame(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

class AprilTagRenderer:
    def __init__(self, video_object: Video, upper_left_corner_tag: str, bottom_right_corner_tag: str, scale: float = 1/8):
        self.video_object = video_object
        self.upper_left_corner_tag_path = upper_left_corner_tag
        self.bottom_right_corner_tag_path = bottom_right_corner_tag
        self.upper_left_corner_tag = None
        self.bottom_right_corner_tag = None 
        self.latest_frame = None
        self.scale = scale
        self._load_tags()

    def _load_tags(self):
        self.upper_left_corner_tag = self._svg_to_numpy(self.upper_left_corner_tag_path)
        self.bottom_right_corner_tag = self._svg_to_numpy(self.bottom_right_corner_tag_path)

    def _svg_to_numpy(self, svg_path):
        with open(svg_path, 'rb') as svg_file:
            svg_data = svg_file.read()
        png_data = cairosvg.svg2png(bytestring=svg_data)
        img = Image.open(io.BytesIO(png_data))
        img = img.convert('RGB')
        return np.array(img)

    def render_tag(self, frame, tag, position, size):
        tag_resized = cv2.resize(tag, (size, size), interpolation=cv2.INTER_AREA)
        
        x, y = position
        frame[y:y+size, x:x+size] = tag_resized
        return frame

    def get_frame(self):
        frame = self.video_object.get_frame()
        height, width = frame.shape[:2]
        
        tag_size = int(min(width, height) * self.scale)
        
        # Render upper left corner tag
        frame = self.render_tag(frame, self.upper_left_corner_tag, (0, 0), tag_size)
        
        # Render bottom right corner tag
        frame = self.render_tag(frame, self.bottom_right_corner_tag, 
                                (width - tag_size, height - tag_size), tag_size)
        
        self.latest_frame = frame
        return frame

if __name__ == "__main__":
    # Create a video object
    debug_video_stream = EmptyVideo(1920, 1080)
    
    # Specify the paths to the AprilTag SVG files
    left_corner_tag = "data/tags/tag41_12_00000.svg"
    right_corner_tag = "data/tags/tag41_12_00001.svg"
    
    # Render April tags on video object
    april_tag_renderer = AprilTagRenderer(debug_video_stream, left_corner_tag, right_corner_tag, scale=1/4)
    
    # Display the rendered frames on a window to the size of the display
    window_name = 'Debug Window'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_width, window_height = 1920, 1080
    cv2.resizeWindow(window_name, window_width, window_height)
    
    last_frame_size = (window_width, window_height)
    while True:
        frame = april_tag_renderer.get_frame()
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Check if window size has changed
        current_frame_size = frame.shape[:2]
        if current_frame_size != last_frame_size:
            window_height, window_width = current_frame_size
            last_frame_size = current_frame_size
            print(f"Window resized to {window_width}x{window_height}")
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()