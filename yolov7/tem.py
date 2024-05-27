import sys
import numpy as np
import io
import os
import cv2  # Import OpenCV for displaying the image

print(os.environ['CONDA_DEFAULT_ENV'])

while True:
    # Read frame size header
    frame_size_header = sys.stdin.buffer.read(4)
    if not frame_size_header:
        break
    
    # Convert header to integer to determine frame size
    frame_size = int.from_bytes(frame_size_header, byteorder='big')
    
    # Read serialized data for frame
    frame_data = sys.stdin.buffer.read(frame_size)
    
    # Deserialize the image
    img_bytes_io = io.BytesIO(frame_data)
    img0 = np.load(img_bytes_io)
    
    # Now you can use img0 for each frame
    print("Received image:", img0.shape)
    
    # Convert image to BGR format (OpenCV's default format)
    img_bgr = cv2.cvtColor(img0.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    
    # Display the image
    cv2.imshow("0", img_bgr)
    cv2.waitKey(1)  # Wait for a short time to allow the image to be displayed
    
# Close OpenCV windows and cleanup
cv2.destroyAllWindows()
