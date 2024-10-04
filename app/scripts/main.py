
import cv2
import numpy as np
import cairosvg
import io
from PIL import Image

class EmptyVideo:
    """
    Empty video object that returns a black frame
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

    def get_latest_frame(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class AprilTagRenderer:
    """
    Renders April tags on a video object to create bounds for a PupilTag surface
    """
    def __init__(self, upper_left_corner_tag: str, bottom_right_corner_tag: str, scale: float = 1/8):

        self.video_object = video_object
        self.upper_left_corner_tag_path = upper_left_corner_tag
        self.bottom_right_corner_tag_path = bottom_right_corner_tag
        self.upper_left_corner_tag = None
        self.bottom_right_corner_tag = None 
        self.latest_frame = None
        self.scale = scale
        self.__load_tags()

    def set_latest_frame(self, frame):
        self.latest_frame = frame

    def get_latest_frame(self):
        frame = self.latest_frame
        height, width = frame.shape[:2]
        
        tag_size = int(min(width, height) * self.scale)
        
        # Render upper left corner tag
        frame = self.__render_tag(frame, self.upper_left_corner_tag, (0, 0), tag_size)
        
        # Render bottom right corner tag
        frame = self.__render_tag(frame, self.bottom_right_corner_tag, 
                                (width - tag_size, height - tag_size), tag_size)
        
        self.latest_frame = frame
        return frame

    
    def __render_tag(self, frame, tag, position, size):
        tag_resized = cv2.resize(tag, (size, size), interpolation=cv2.INTER_AREA)
        
        x, y = position
        frame[y:y+size, x:x+size] = tag_resized
        return frame

    def __load_tags(self):
        self.upper_left_corner_tag = self.__svg_to_numpy(self.upper_left_corner_tag_path)
        self.bottom_right_corner_tag = self.__svg_to_numpy(self.bottom_right_corner_tag_path)

    def __svg_to_numpy(self, svg_path):

        with open(svg_path, 'rb') as svg_file:
            svg_data = svg_file.read()
        png_data = cairosvg.svg2png(bytestring=svg_data)
        img = Image.open(io.BytesIO(png_data))
        img = img.convert('RGB')
        return np.array(img)



class SurfaceListener():
    """
    Networking class to listen for PupilTag surface data
    """
    def __init__(self, ip='localhost', port=50020):
        try:
            self.ip = ip
            self.port = port
            self.context = zmq.Context()
            self.pupil_remote = self.context.socket(zmq.REQ)
            self.sub_port = None
            self.subscriber = None

            self.surface_topic_name = 'surface'
            self.gaze_key = 'gaze_on_surfaces'
            self.fixation_key = 'fixations_on_surfaces'


            self.latest_gaze_coordinates = (None, None)
            self.latest_fixation_coordinates = (None, None)

            self.__connect_to_pupil()

        except Exception as e:
            print(f"SurfaceListener: Error connecting to Pupil: {e}")
            raise e


    def get_latest_filtered_gaze_coordinates(self):
        """
        output: list of tuples of (x, y) coordinates normalized to [1,0], returns (None, None) if no latest gaze data
        """
        try:
            topic, payload = subscriber.recv_multipart()
            message = msgpack.loads(payload)

            for key, value in message.items():
                if key == 'gaze_on_surfaces':
                    for gaze_data in value:
                        if 'norm_pos' in gaze_data:
                            x, y = gaze_data['norm_pos']
                            return x, y

        except Exception as e:
            raise e


    def get_latest_filtered_fixation_coordinates(self):
        """
        output: list of tuples of (x, y) coordinates normalized to [1,0], returns (None, None) if no latest fixation data
        """
        try:
            pass

        except Exception as e:
            pass        

    def __connect_to_pupil(self):
        try:
            self.pupil_remote.connect(f'tcp://{self.ip}:{self.port}')
            self.pupil_remote.send_string('SUB_PORT')
            self.sub_port = self.pupil_remote.recv_string()

            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.connect(f'tcp://{self.ip}:{self.sub_port}')

            self.subscriber.subscribe(self.surface_topic_name)


        except Exception as e:
            raise e



class FrameDrawing():
    """
    Utility class to demo surface listening. 
    Get's surface gaze data to create heatmap on Video object, fixations are used to draw stars on the correcposing Video object point
    
    Outputs the rendered frame. 
    Note: if no gaze data is available, the frame will be unchanged
    """
    def __init__(self):
        pass

    def set_latest_frame(self, frame):
        pass

    def get_latest_frame(self):
        pass



if __name__ == "__main__":
    try:
        # establish connection
        surface_listener = SurfaceListener()

        # Create a video object
        debug_video_stream = EmptyVideo(1920, 1080)
        
        # Specify the paths to the AprilTag SVG files
        left_corner_tag = "tags/tag41_12_00000.svg"
        right_corner_tag = "tags/tag41_12_00001.svg"
        
        # Create AprilTagRenderer object
        april_tag_renderer = AprilTagRenderer(left_corner_tag, right_corner_tag, scale=1/4)


        # Create FrameDrawing object
        frame_drawer = FrameDrawing()


        # Display the rendered frames on a window to the size of the display
        window_name = 'Debug Window'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_width, window_height = 1920, 1080
        cv2.resizeWindow(window_name, window_width, window_height)

        while True:
            # Get raw frame
            raw_frame = debug_video_stream.get_latest_frame()
            
            # Pass frame to Drawer, get output
            frame_drawer.set_latest_frame(raw_frame, 
            surface_listener.get_latest_filtered_gaze_coordinates(), surface_listener.get_latest_filtered_fixation_coordinates())


            # Add April tags to frame and output the frame

            april_tag_renderer.get_frame()
            

            # Output the layers to a window

            cv2.imshow(window_name, final_frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        cv2.destroyAllWindows()
        surface_listener.close()

    except Exception as e:
        print(f"Application Error: {e}")
        cv2.destroyAllWindows()
        surface_listener.close()
        raise e