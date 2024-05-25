"""
Quantization – reduces precision of transformed data
DCT itself is lossless, and its purpose is to transform the image from the spatial domain to the frequency domain.
To achieve compression, we need to introduce a step between DCT and IDCT.
The quantization matrix values can be scaled by a quantization parameter (QP).
Lower QP values result in more aggressive quantization (higher compression) and vice versa.
References
[1] Understanding DCT and Quantization in JPEG compression, 
https://dev.to/marycheung021213/understanding-dct-and-quantization-in-jpeg-compression-1col
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# 3.b, 3.c is valid only for C++, Python allocates memory automatically
class ImageProcessorApp:
    def __init__(self, root):
        self.N = 8     # 2.a, 3.a
        self.QP = 32

        self.image = None
        self.root = root
        self.root.title("Image Processor")

        self.original_frame = tk.Frame(root)
        self.original_frame.pack(side="left")

        self.processed_frame = tk.Frame(root)
        self.processed_frame.pack(side="left")

        self.original_label = tk.Label(self.original_frame)
        self.original_label.pack()

        self.processed_label = tk.Label(self.processed_frame)
        self.processed_label.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side="top", anchor="nw", padx=10, pady=10)  # Keep buttons at top left

        self.open_button = tk.Button(self.button_frame, text="Open", command=self.open_image)
        self.open_button.pack(side="left", padx=5, pady=5)

        self.dct_button = tk.Button(self.button_frame, text="RLE Decode/Encode", command=self.perform_dct)
        self.dct_button.pack(side="left", padx=5, pady=5)

        self.image_path = None

    def open_image(self):
        # 3.e When "Open" button is used, the original image is shown on GUI
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            self.show_original_image()

    def show_original_image(self):
        # 2.b original_image variable to store original image
        original_image = cv2.imread(self.image_path)
        modified_img = np.copy(original_image)
        grayscale_image = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)
        processed_image = Image.fromarray(grayscale_image)
        img = ImageTk.PhotoImage(processed_image)
        self.original_label.configure(image=img)
        self.original_label.image = img
    
    def perform_dct(self):
        print("Perform DCT")
        if self.image_path:
            # Example: Process image using OpenCV
            image = cv2.imread(self.image_path)
            height, width, channels = image.shape
            blocks = 0
            if width >= height: # Make sure we don't try to iterate outside the image
                blocks = int(width / self.N)
            else:
                blocks = int(height / self.N)
            modified_img = np.copy(image)
            grayscale_image = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)  # Example: Convert to grayscale
            self.dct_image = self.calcDCT(grayscale_image, blocks)
            # zigZagArray = self.zigzag_scan(self.dct_image)

            # Perform image processing here
            # 3.d, 4 Display processed grayscale image, store it in a new variable
            # 4 The processed image will be derived from original image, so no duplication needed in Python
            processed_image = Image.fromarray(self.dct_image)
            processed_image = ImageTk.PhotoImage(processed_image)
            self.processed_label.configure(image=processed_image)
            self.processed_label.image = processed_image
    
    def perform_idct(self):
        print("Perform IDCT")
        processed_image = Image.fromarray(self.reconstructed_image)
        processed_image = ImageTk.PhotoImage(processed_image)
        self.processed_label.configure(image=processed_image)
        self.processed_label.image = processed_image
    
    def calcDCT(self, grayscale_image, blocks):
        dct_image = grayscale_image
        self.reconstructed_image = grayscale_image
        for rowIndex in range(0, blocks):
            for columnIndex in range(0, blocks):
                row = rowIndex * self.N
                column = columnIndex * self.N
                # Extract a block of NxN pixels from the image
                block = grayscale_image[row:row+self.N, column:column+self.N]
                # print(block)
                # cv2.imshow('Block', block)
                # Compute the DCT
                dct_block = cv2.dct(np.float32(block), cv2.DCT_INVERSE)
                # Quantization
                FQ = np.around(dct_block / self.QP)
                Y = FQ * self.QP
                np.savetxt('doc/L5/blocksize{0}_QP{1}.txt'.format(self.N, self.QP), Y, fmt='%d')
                #dct_image[row:row+self.N, column:column+self.N] = iq_coeff
                print(Y)
                dct_image[row:row+self.N, column:column+self.N] = Y
                D1 = self.zigzag_scan(Y)
                print("Zig-Zag Scanning - 2D -> 1D Array: " + str(len(D1)))
                print(D1)
                np.savetxt('doc/L5/zig_zag_scan{0}_QP{1}.csv'.format(self.N, self.QP), D1, delimiter=',', fmt='%d')
                rle_encode = self.rle_encode(D1)
                print("RLE Encoding")
                print(rle_encode)
                with open('doc/L5/RLE_encoding{0}_QP{1}.csv'.format(self.N, self.QP), 'a') as f:
                    np.savetxt(f, rle_encode, delimiter=',', fmt='%d')
                    f.flush()
                    np.savetxt(f, ["EOB"], delimiter=',', fmt='%s')
                    f.flush()
                
                rle_decode = self.rle_decode(rle_encode)
                print("RLE Decoding")
                print(rle_decode)
                dct_decoded = self.reverse_zigzag_scan(rle_decode, self.N, self.N)
                print("DCT Decoded")
                print(dct_decoded)
                #import time
                #time.sleep(1000000000)
                # IDCT
                #idct_block = cv2.idct(iq_coeff)
                idct_block = cv2.idct(dct_decoded)
                self.reconstructed_image[row:row+self.N, column:column+self.N] = np.uint8(idct_block)
                # print(dct_block)
                # Display the result
                # cv2.imshow('DCTBlock', dct_block)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        self.dct_img = np.copy(dct_image)
        return dct_image

    def calcIDCT(self, dct_image, blocks):
        reconstructed_image = dct_image
        for rowIndex in range(0, blocks):
            for columnIndex in range(0, blocks):
                row = rowIndex * self.N
                column = columnIndex * self.N
                # Extract a block of NxN pixels from the image
                block = self.dct_img[row:row+self.N, column:column+self.N]
                idct_block = cv2.idct(block)
                reconstructed_image[row:row+self.N, column:column+self.N] = np.uint8(idct_block)
        return reconstructed_image
    
    def zigzag_scan(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        result = []
        row, col = 0, 0

        for _ in range(rows * cols):
            result.append(matrix[row][col])

            if (row + col) % 2 == 0:  # Even sum of indices
                if col == cols - 1:
                    row += 1
                elif row == 0:
                    col += 1
                else:
                    row -= 1
                    col += 1
            else:  # Odd sum of indices
                if row == rows - 1:
                    col += 1
                elif col == 0:
                    row += 1
                else:
                    row += 1
                    col -= 1
        return result
    
    def reverse_zigzag_scan(self, array, rows, cols):
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        matrix = np.zeros((rows, cols), dtype=array.dtype)
        row, col = 0, 0

        for i in range(rows * cols):
            matrix[row, col] = array[i]

            if (row + col) % 2 == 0:  # Even sum of indices
                if col == cols - 1:
                    row += 1
                elif row == 0:
                    col += 1
                else:
                    row -= 1
                    col += 1
            else:  # Odd sum of indices
                if row == rows - 1:
                    col += 1
                elif col == 0:
                    row += 1
                else:
                    row += 1
                    col -= 1

        return matrix
    
    def rle_encode(self, data):
        if not data:
            return []

        encoding = []
        zeroCount = 0.0
        for char in data[0:]:
            if char != 0.0 or char != -0.0:
                encoding.append((zeroCount, char))
                zeroCount = 0.0   # Reset counter
            else:
                zeroCount += 1  # Increment 0 counter

        return encoding

    def rle_decode(self, encoded_data):
        decoded_data = []
    
        for zeroCount, char in encoded_data:
            for i in range(int(zeroCount)):
                decoded_data.append(0.0)
            decoded_data.append(char)
        
        # Fill in remaining zeros to account for an array size of NxN
        for i in range(self.N * self.N - len(decoded_data)):
            decoded_data.append(0.0)
        
        return decoded_data

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()