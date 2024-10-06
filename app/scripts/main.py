import cv2
import numpy as np
import cairosvg
import io
from PIL import Image
import zmq
import msgpack
import logging
import argparse
import time
from typing import Tuple, Optional
from datetime import datetime

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


class SurfaceListener:
    def __init__(self, ip='127.0.0.1', port=50020, initial_connection_timeout=5, message_timeout=1):

        self.logger = logging.getLogger(__name__)
        self.ip = ip
        self.port = port
        self.initial_connection_timeout = initial_connection_timeout
        self.message_timeout = message_timeout

        self.context = zmq.Context()
        self.pupil_remote = None
        self.subscriber = None

        self.surface_topic_name = 'surfaces'.encode('utf-8')
        self.gaze_key = 'gaze_on_surfaces'
        self.fixation_key = 'fixations_on_surfaces'

        self.latest_gaze_coordinates = (None, None)
        self.latest_fixation_coordinates = (None, None)

        # self.__connect_to_pupil()

    def __connect_to_pupil(self):
        try:
            self.pupil_remote = self.context.socket(zmq.REQ)
            self.pupil_remote.setsockopt(zmq.LINGER, 0)
            self.pupil_remote.connect(f'tcp://{self.ip}:{self.port}')
            
            # Send a request for the sub port
            self.pupil_remote.send_string('SUB_PORT')
            self.sub_port = self.pupil_remote.recv_string()

            self.subscriber = self.context.socket(zmq.SUB)
            self.subscriber.setsockopt(zmq.LINGER, 0)
            self.subscriber.connect(f'tcp://{self.ip}:{self.sub_port}')

            self.logger.info("Connected to Pupil")

            self.__connect_to_surface_topic()

        except (zmq.ZMQError, TimeoutError) as e:
            self.logger.warning(f"Error connecting to Pupil: {e}")
            self.pupil_remote = None

        except Exception as e:
            self.logger.warning(f"Unexpected error connecting to Pupil: {e}")
            self.pupil_remote = None

    def __connect_to_surface_topic(self):
        try:
            self.subscriber.subscribe(self.surface_topic_name)
            self.logger.info("Connected to surface topic")
        except (zmq.ZMQError, TimeoutError) as e:
            self.logger.warning(f"Error connecting to surface topic: {e}")
        except Exception as e:
            self.logger.warning(f"Unexpected error connecting to surface topic: {e}")

    def get_latest_surface_data(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if self.pupil_remote is None:
            return None, None, None, None

        if not self.subscriber:
            self.logger.debug("Surface topic was not available. Attempting to reconnect...")
            self.__connect_to_surface_topic()
            if not self.subscriber:
                self.logger.warning("Unable to connect to surface topic")
                return None, None, None, None
            else:
                self.logger.info("Reconnected to surface topic")

        try:
            if self.subscriber.poll(timeout=self.message_timeout * 1000) == 0:
                self.logger.debug("No new surface data received within timeout period")
                return None, None, None, None

            topic, payload = self.subscriber.recv_multipart(zmq.NOBLOCK)
            message = msgpack.loads(payload)

            if not message.get('surfaces'):
                self.logger.debug("No surfaces detected")
                return None, None, None, None

            for surface in message['surfaces']:
                gaze_on_surf = surface.get(self.gaze_key, [])
                fixations_on_surf = surface.get(self.fixation_key, [])

                if gaze_on_surf:
                    self.latest_gaze_coordinates = gaze_on_surf[-1].get('norm_pos', (None, None))
                
                if fixations_on_surf:
                    self.latest_fixation_coordinates = fixations_on_surf[-1].get('norm_pos', (None, None))

            return (self.latest_gaze_coordinates[0], self.latest_gaze_coordinates[1],
                    self.latest_fixation_coordinates[0], self.latest_fixation_coordinates[1])

        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                self.logger.debug("No new surface data available")
            else:
                self.logger.warning(f"Error getting surface data: {e}")
        except Exception as e:
            self.logger.warning(f"Unexpected error getting surface data: {e}")

        return None, None, None, None

    def close(self):
        if self.pupil_remote:
            self.pupil_remote.close()
        if self.subscriber:
            self.subscriber.close()
        self.context.term()
        self.logger.info("Closed ZMQ connections")

class FrameDrawing():
    """
    Utility class to demo surface listening. 
    Get's surface gaze data to create heatmap on Video object, 
    fixations are used to draw small circles on the correcposing Video object point
    
    Outputs the rendered frame. 
    Note: if no gaze data is available, the frame will be unchanged
    """
    def __init__(self):
        self.latest_frame = None
        self.frame_width = None
        self.frame_height = None

        self.gaze_heatmap = None
        self.fixation_layer = None
        self.decay_factor = 0.95
        self.max_intensity = 255

    def set_latest_frame(self, frame, gaze_coordinates, fixation_coordinates):
        self.latest_frame = frame
        self.frame_width, self.frame_height = frame.shape[:2]
        
        self.update_gaze_heatmap(gaze_coordinates)
        self.update_fixation_layer(fixation_coordinates)

    def get_latest_frame(self):

        # If no gaze or fixation data (Likely the glasses not connected) then just return the input latest frame
        if self.gaze_heatmap is None and self.fixation_layer is None:
            return self.latest_frame

        else:
            # Apply heatmap
            heatmap = cv2.applyColorMap((self.gaze_heatmap * self.max_intensity).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_layer = cv2.addWeighted(base_frame, 0.7, heatmap, 0.3, 0)

            # Apply fixation layer
            mask = self.fixation_layer[:, :, 3] / 255.0
            mask = np.stack([mask] * 3, axis=-1)
            final_frame = heatmap_layer * (1 - mask) + self.fixation_layer[:, :, :3] * mask

            return final_frame.astype(np.uint8)

    def update_gaze_heatmap(self, gaze_coordinates):
        if gaze_coordinates[0] is not None and gaze_coordinates[1] is not None:
            x, y = int(gaze_coordinates[0] * self.frame_width), int(gaze_coordinates[1] * self.frame_height)
            cv2.circle(self.gaze_heatmap, (x, y), 20, (1, 1, 1), -1)
        
            self.gaze_heatmap *= self.decay_factor
            self.gaze_heatmap = np.clip(self.gaze_heatmap, 0, 1)

        else:
            return


    def update_fixation_layer(self, fixation_coordinates):
        if fixation_coordinates[0] is not None and fixation_coordinates[1] is not None:
            x, y = int(fixation_coordinates[0] * self.frame_width), int(fixation_coordinates[1] * self.frame_height)
            cv2.circle(self.fixation_layer, (x, y), 5, (0, 255, 0, 255), -1)

    def clear_drawing(self):
        self.gaze_heatmap.fill(0)
        self.fixation_layer.fill(0)


def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"gaze_app_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=f"logs/{log_filename}",
        filemode='w'
    )
    
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze Drawing App")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("Starting Gaze Drawing App")

    try:
        # Establish connection
        surface_listener = SurfaceListener()
        logger.info("Attempted connection to Surface Listener")

        # Create a video object
        debug_video_stream = EmptyVideo(1920, 1080)
        logger.info("Created Empty Video stream")

        # Specify the paths to the AprilTag SVG files
        left_corner_tag = "tags/tag41_12_00000.svg"
        right_corner_tag = "tags/tag41_12_00001.svg"
        
        # Create AprilTagRenderer object
        april_tag_renderer = AprilTagRenderer(left_corner_tag, right_corner_tag, scale=1/4)
        logger.info("Created AprilTagRenderer")

        # Create FrameDrawing object
        frame_drawer = FrameDrawing()
        logger.info("Created FrameDrawing object")

        # Display the rendered frames on a window to the size of the display
        window_name = 'Debug Window'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_width, window_height = 1920, 1080
        cv2.resizeWindow(window_name, window_width, window_height)
        logger.info(f"Created display window: {window_name}")

        frame_count = 0
        while True:
            # Get raw frame
            raw_frame = debug_video_stream.get_latest_frame()
            
            # Get gaze and fixation coordinates
            gaze_x, gaze_y, fixation_x, fixation_y = surface_listener.get_latest_surface_data()
            
            # Pass frame to Drawer, get output
            frame_drawer.set_latest_frame(raw_frame, (gaze_x, gaze_y), (fixation_x, fixation_y))
            drawn_frame = frame_drawer.get_latest_frame()

            # Add April tags to frame and output the frame
            april_tag_renderer.set_latest_frame(drawn_frame)
            final_frame = april_tag_renderer.get_latest_frame()

            # Output the layers to a window
            cv2.imshow(window_name, final_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:  # Log every 100 frames
                logger.debug(f"Processed frame {frame_count}")
                logger.debug(f"Got raw frame: {raw_frame.shape}")
                logger.debug(f"Got gaze coordinates: ({gaze_x}, {gaze_y})")
                logger.debug(f"Got fixation coordinates: ({fixation_x}, {fixation_y})")
                logger.debug(f"Got drawn frame: {drawn_frame.shape}")
                logger.debug(f"Got final frame: {final_frame.shape}")
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit signal received")
                break
        
        # Close the window if loop is exited
        cv2.destroyAllWindows()
        surface_listener.close()
        logger.info("Application closed successfully")

    except Exception as e:
        logger.exception(f"Application Error: {e}")
        cv2.destroyAllWindows()
        if 'surface_listener' in locals():
            surface_listener.close()
        raise e