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
import threading
from collections import deque
import queue
from logging.handlers import QueueHandler, QueueListener

def setup_logging(debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"gaze_app_{timestamp}.log"
    
    # Create a queue for log messages
    log_queue = queue.Queue(-1)  # no limit on size

    # Setup handlers
    file_handler = logging.FileHandler(filename=f"logs/{log_filename}", mode='w')
    console_handler = logging.StreamHandler()
    
    # Setup formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Setup the root logger
    root = logging.getLogger()
    root.setLevel(log_level)
    root.addHandler(QueueHandler(log_queue))

    # Start the listener in a separate thread
    listener = QueueListener(log_queue, file_handler, console_handler)
    listener.start()

    return listener  # Return the listener so it can be stopped later


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
    def __init__(self, surface_name="Surface 1", ip='127.0.0.1', port=50020, reconnect_interval=5, buffer_size=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ip = ip
        self.port = port
        self.reconnect_interval = reconnect_interval
        self.buffer_size = buffer_size
        self.surface_name = surface_name

        self.context = zmq.Context()
        self.pupil_remote = None
        self.subscriber = None
        self.sub_port = None

        self.topic = 'surfaces'

        self.data_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        self.logger.debug("SurfaceListener thread started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self._close_connections()
        self.logger.debug("SurfaceListener stopped")

    def _run(self):
        while self.running:
            if not self._ensure_connection():
                self.logger.debug(f"Connection failed, retrying in {self.reconnect_interval} seconds")
                time.sleep(self.reconnect_interval)
                continue

            self._receive_data()

    def _ensure_connection(self):
        if self.pupil_remote is None or self.subscriber is None:
            try:
                self._connect_to_pupil()
                return True
            except Exception as e:
                self.logger.warning(f"Connection failed: {e}")
                return False
        return True

    def _connect_to_pupil(self):
        self.pupil_remote = self.context.socket(zmq.REQ)
        self.pupil_remote.connect(f'tcp://{self.ip}:{self.port}')

        self.pupil_remote.send_string('SUB_PORT')
        self.sub_port = self.pupil_remote.recv_string()

        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(f'tcp://{self.ip}:{self.sub_port}')
        self.subscriber.subscribe(self.topic)

        self.logger.info(f"Connected to Pupil on port {self.port}, subscribed on port {self.sub_port}")
        self.logger.info(f"Subscribed to topic: {self.topic}")

    def _receive_data(self):
        try:
            topic, payload = self._recv_from_sub()
            if topic is None or payload is None:
                return

            if payload.get('name') == self.surface_name:
                self._process_surface_data(payload)

        except Exception as e:
            self.logger.warning(f"Unexpected error getting data: {e}")
            self._close_connections()

    def _recv_from_sub(self):
        try:
            topic = self.subscriber.recv_string(zmq.NOBLOCK)
            payload = msgpack.unpackb(self.subscriber.recv(zmq.NOBLOCK), raw=False)
            return topic, payload
        except zmq.Again:
            return None, None
        except Exception as e:
            self.logger.warning(f"Error in _recv_from_sub: {e}")
            return None, None

    def _process_surface_data(self, payload):
        gaze_data = payload.get('gaze_on_surfaces', [])
        fixation_data = payload.get('fixations_on_surfaces', [])

        if gaze_data:
            gaze_x, gaze_y = gaze_data[-1].get('norm_pos', (None, None))
        else:
            gaze_x, gaze_y = None, None

        if fixation_data:
            fixation_x, fixation_y = fixation_data[-1].get('norm_pos', (None, None))
        else:
            fixation_x, fixation_y = None, None

        self._add_to_buffer(gaze_x, gaze_y, fixation_x, fixation_y)

    def _add_to_buffer(self, gaze_x, gaze_y, fixation_x, fixation_y):
        with self.lock:
            self.data_buffer.append((gaze_x, gaze_y, fixation_x, fixation_y))

    def get_latest_surface_data(self):
        with self.lock:
            return self.data_buffer[-1] if self.data_buffer else (None, None, None, None)

    def _close_connections(self):
        if self.pupil_remote:
            self.pupil_remote.close()
            self.pupil_remote = None
        if self.subscriber:
            self.subscriber.close()
            self.subscriber = None
        self.logger.info("Closed ZMQ connections")


class FrameDrawing:
    """
    Utility class to demo surface listening. 
    Get's surface gaze data to create heatmap on Video object, 
    fixations are used to draw small circles on the correcposing Video object point
    
    Outputs the rendered frame. 
    Note: if no gaze data is available, the frame will be unchanged
    """
    def __init__(self, width=1920, height=1080):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.latest_frame = None
        self.frame_width = width
        self.frame_height = height

        self.gaze_heatmap = np.zeros((height, width), dtype=np.float32)
        self.fixation_layer = np.zeros((height, width, 4), dtype=np.uint8)
        self.decay_factor = 0.95
        self.max_intensity = 255

    def set_latest_frame(self, frame, gaze_coordinates, fixation_coordinates):
        if frame.shape[:2] != (self.frame_height, self.frame_width):
            self.logger.warning(f"Frame size mismatch. Expected {self.frame_width}x{self.frame_height}, got {frame.shape[1]}x{frame.shape[0]}")
            self.frame_height, self.frame_width = frame.shape[:2]
            self.gaze_heatmap = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
            self.fixation_layer = np.zeros((self.frame_height, self.frame_width, 4), dtype=np.uint8)

        self.latest_frame = frame
        self.update_gaze_heatmap(gaze_coordinates)
        self.update_fixation_layer(fixation_coordinates)

    def get_latest_frame(self):
        if self.latest_frame is None:
            self.logger.warning("No frame available")
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Apply heatmap
        heatmap = cv2.applyColorMap((self.gaze_heatmap * self.max_intensity).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_layer = cv2.addWeighted(self.latest_frame, 0.7, heatmap, 0.3, 0)

        # Apply fixation layer
        mask = self.fixation_layer[:, :, 3] / 255.0
        mask = np.stack([mask] * 3, axis=-1)
        final_frame = heatmap_layer * (1 - mask) + self.fixation_layer[:, :, :3] * mask

        return final_frame.astype(np.uint8)

    def update_gaze_heatmap(self, gaze_coordinates):
        if gaze_coordinates[0] is not None and gaze_coordinates[1] is not None:
            x = int(gaze_coordinates[0] * self.frame_width)
            y = int((1 - gaze_coordinates[1]) * self.frame_height)  # Invert Y coordinate
            
            # Ensure coordinates are within the frame
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                cv2.circle(self.gaze_heatmap, (x, y), 20, (1, 1, 1), -1)
                self.gaze_heatmap *= self.decay_factor
                self.gaze_heatmap = np.clip(self.gaze_heatmap, 0, 1)
            else:
                self.logger.warning(f"Gaze coordinates ({x}, {y}) out of frame bounds")

    def update_fixation_layer(self, fixation_coordinates):
        if fixation_coordinates[0] is not None and fixation_coordinates[1] is not None:
            x = int(fixation_coordinates[0] * self.frame_width)
            y = int((1 - fixation_coordinates[1]) * self.frame_height)  # Invert Y coordinate
            
            # Ensure coordinates are within the frame
            if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
                cv2.circle(self.fixation_layer, (x, y), 5, (0, 255, 0, 255), -1)
            else:
                self.logger.warning(f"Fixation coordinates ({x}, {y}) out of frame bounds")
    def clear_drawing(self):
        self.gaze_heatmap.fill(0)
        self.fixation_layer.fill(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaze Drawing App")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_listener = setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    logger.info("Starting Gaze Drawing App")

    try:
        # Establish connection
        surface_listener = SurfaceListener()
        surface_listener.start()
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
        frame_drawer = FrameDrawing(1920, 1080)
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

            if gaze_x is not None and gaze_y is not None:
                logger.debug(f"Got gaze coordinates: ({gaze_x}, {gaze_y})")

            if fixation_x is not None and fixation_y is not None:
                logger.debug(f"Got fixation coordinates: ({fixation_x}, {fixation_y})")
            
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
        surface_listener.stop()
        logger.info("Application closed successfully")

    except Exception as e:
        logger.exception(f"Application Error: {e}")
        cv2.destroyAllWindows()
        if 'surface_listener' in locals():
            surface_listener.stop()
        raise e

    finally:
        log_listener.stop()