
import cv2
import numpy as np

# Global variables to store window size
window_width, window_height = 640, 480
last_frame_size = (window_width, window_height)

def draw_static_elements(frame):
    height, width = frame.shape[:2]
    
    # Center circle
    center = (width // 2, height // 2)
    radius = min(width, height) // 20  # Adjust size relative to window
    cv2.circle(frame, center, radius, (0, 255, 0), -1)
    
    # Squares in corners
    square_size = min(width, height) // 15  # Adjust size relative to window
    
    # Top-left
    cv2.rectangle(frame, (0, 0), (square_size, square_size), (255, 0, 0), -1)
    
    # Top-right
    cv2.rectangle(frame, (width - square_size, 0), (width, square_size), (255, 0, 0), -1)
    
    # Bottom-left
    cv2.rectangle(frame, (0, height - square_size), (square_size, height), (255, 0, 0), -1)
    
    # Bottom-right
    cv2.rectangle(frame, (width - square_size, height - square_size), (width, height), (255, 0, 0), -1)

def main():
    global window_width, window_height, last_frame_size
    window_name = 'Resizable Window with Static Elements'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    while True:
        # Create a black canvas
        frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Draw static elements
        draw_static_elements(frame)
        
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

if __name__ == "__main__":
    main()