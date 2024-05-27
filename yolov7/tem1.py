import sys
import numpy as np
import io
import os
# Generate your images
num_frames = 5
for i in range(num_frames):
    # Generate image
    img = np.zeros((192, 168, 3), dtype=np.uint8)  # Example image
    
    # Serialize the image
    img_bytes_io = io.BytesIO()
    np.save(img_bytes_io, img)
    img_bytes = img_bytes_io.getvalue()
    
    # Send frame size as a header
    sys.stdout.buffer.write(len(img_bytes).to_bytes(4, byteorder='big'))
    
    # Send serialized data for frame
    sys.stdout.buffer.write(img_bytes)

    # Flush stdout buffer to ensure data is sent immediately
    sys.stdout.flush()
