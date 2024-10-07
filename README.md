# Pupil Surface Tracking with Digital AprilTag

This repository contains an example of how to use the Pupil Labs eye tracking system to track a surface with a digital AprilTag. The system uses the Pupil Capture software to detect the AprilTag and the Pupil Service to filter the surface data. The filtered surface data can then be subscribed to via ZMQ.

## Setup

1. Build the Docker Images: `docker-compose build`
2. Run the Docker Containers: `docker-compose up`
    Note: You may need to enable X11 forwarding on your host machine: `xhost +local:root`
3. In the Pupil Capture, calibrate your eye tracker.
4. In the Pupil Capture, setup your surface with apriltag. This app uses the tag family "tag41h12". The tags 0 and 1 should be detected. Update the other corners of the surface to bound your surface.
5. Setup a surface with tag family "tag41h12", the tags 0 and 1 should be detected. Update the other corners of the surface to bound your surface.
6. Close the Pupil Capture window. Then you can stop the containers. 
    Note: If you stop the containers without first closing the window, the Pupil Capture will not save the calibration data and you will have no gaze data on the next launch.

The surface should now be tracked and gaze data available. You can access filtered surface data by subscribing to the "surface" topic via ZMQ. An examples is shown below:

```python
import zmq
import msgpack

def gaze_listener(drawing_app):
    ctx = zmq.Context()
    pupil_remote = ctx.socket(zmq.REQ)
    ip = 'localhost'
    port = 50020
    pupil_remote.connect(f'tcp://{ip}:{port}')

    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()

    subscriber = ctx.socket(zmq.SUB)
    subscriber.connect(f'tcp://{ip}:{sub_port}')
    subscriber.subscribe('surface')

    while True:
        topic, payload = subscriber.recv_multipart()
        message = msgpack.loads(payload)
        
        for key, value in message.items():
            if key == 'gaze_on_surfaces':
                for gaze_data in value:
                    if 'norm_pos' in gaze_data:
                        x, y = gaze_data['norm_pos']
```