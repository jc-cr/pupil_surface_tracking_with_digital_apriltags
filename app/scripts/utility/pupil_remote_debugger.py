import zmq
import time
import logging
from datetime import datetime
import os
import msgpack

def setup_logging(debug=True):
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pupil_debug_{timestamp}.log"
    log_dir = "logs"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(log_dir, log_filename),
        filemode='w'
    )
    
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)

def recv_from_sub(socket, timeout=1000):
    try:
        topic = socket.recv_string(flags=zmq.NOBLOCK)
        payload = socket.recv(flags=zmq.NOBLOCK)
        try:
            # Try to decode as string
            payload = payload.decode('utf-8')
        except UnicodeDecodeError:
            # If it fails, treat as binary data
            try:
                payload = msgpack.unpackb(payload)
            except Exception as e:
                payload = f"<Binary data: unable to unpack, error: {str(e)}>"
        return topic, payload
    except zmq.Again:
        return None, None

def test_pupil_connection(ip, port, logger):
    context = zmq.Context()
    
    # Connect to Pupil Remote
    pupil_remote = context.socket(zmq.REQ)
    pupil_remote.connect(f"tcp://{ip}:{port}")
    
    logger.info(f"Connected to Pupil Remote at {ip}:{port}")

    # Request sub port
    pupil_remote.send_string("SUB_PORT")
    sub_port = pupil_remote.recv_string()
    logger.info(f"Subscriber port: {sub_port}")

    # Connect to sub socket
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{ip}:{sub_port}")
    sub_socket.subscribe("")  # Subscribe to all topics

    # Request available topics
    pupil_remote.send_string("PUB_SOCKET")
    pub_socket = pupil_remote.recv_string()
    logger.info(f"Pub Socket: {pub_socket}")

    try:
        while True:
            # Check for new messages
            topic, payload = recv_from_sub(sub_socket)
            if topic is not None:
                logger.info(f"Received topic: {topic}")
                logger.debug(f"Payload: {payload}")

            # Request topics every 5 seconds
            pupil_remote.send_string("t")
            pupil_time = pupil_remote.recv_string()
            logger.info(f"Pupil time: {pupil_time}")

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        pupil_remote.close()
        sub_socket.close()
        context.term()

if __name__ == "__main__":
    logger = setup_logging(debug=True)
    test_pupil_connection('127.0.0.1', '50020', logger)