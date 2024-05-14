"""
Quantization – reduces precision of transformed data
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
        self.N = 256     # 2.a, 3.a

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

        self.dct_button = tk.Button(self.button_frame, text="DCT", command=self.perform_dct)
        self.dct_button.pack(side="left", padx=5, pady=5)

        self.idct_button = tk.Button(self.button_frame, text="IDCT", command=self.perform_idct)
        self.idct_button.pack(side="left", padx=5, pady=5)

        self.image_path = None

    def open_image(self):
        # 3.e When "Open" button is used, the original image is shown on GUI
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            self.show_original_image()

    def show_original_image(self):
        # 2.b original_image variable to store original image
        original_image = Image.open(self.image_path)
        original_image = ImageTk.PhotoImage(original_image)
        self.original_label.configure(image=original_image)
        self.original_label.image = original_image
    
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
                print(block)
                cv2.imshow('Block', block)
                # Compute the DCT
                dct_block = cv2.dct(np.float32(block), cv2.DCT_INVERSE)
                dct_image[row:row+self.N, column:column+self.N] = dct_block
                idct_block = cv2.idct(dct_block)
                self.reconstructed_image[row:row+self.N, column:column+self.N] = np.uint8(idct_block)
                print(dct_block)
                # Display the result
                cv2.imshow('DCTBlock', dct_block)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        print(dct_image)
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()