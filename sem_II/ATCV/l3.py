import numpy as np
import cv2

"""
In Discrete Cosine Transform (DCT) compression, the input block is transformed into the frequency domain using DCT.
For a 4x4 block with all values 128, the DCT coefficients can be calculated directly in Python with cv2.dct().
Since all values in the block are the same, the DCT coefficients will contain mostly low-frequency components.
"""

# Define the 4x4 image block with all values 128
block = np.full((4, 4), 128, dtype=np.float32)
print(block)

# Apply DCT to the image block
dct_block = cv2.dct(block)

# Print the DCT coefficients
print("DCT coefficients:")
print(dct_block)
