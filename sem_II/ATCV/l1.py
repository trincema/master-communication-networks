import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessorApp:
    def __init__(self, root):
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

        self.process_button = tk.Button(self.button_frame, text="Process", command=self.process_image)
        self.process_button.pack(side="left", padx=5, pady=5)

        self.image_path = None

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if self.image_path:
            self.show_original_image()

    def show_original_image(self):
        original_image = Image.open(self.image_path)
        original_image = ImageTk.PhotoImage(original_image)
        self.original_label.configure(image=original_image)
        self.original_label.image = original_image
    
    def modify_brightness(self, img, x, y, size, brightness_change):
        """
        Modify the brightness of a region in the image.

        Parameters:
            img: Input image.
            x, y: Top-left coordinates of the region.
            size: Size of the region (width and height).
            brightness_change: Amount to change the brightness (positive or negative).

        Returns:
            Modified image.
        """
        # Extract the region of interest
        roi = img[y:y+size, x:x+size]
        # Change the brightness of the region
        roi = cv2.add(roi, brightness_change)
        # Clip the pixel values to the valid range [0, 255]
        roi = np.clip(roi, 0, 255)
        # Replace the region in the modified image
        img[y:y+size, x:x+size] = roi
        return img

    def process_image(self):
        image = cv2.imread(self.image_path)
        # Create a copy of the input image and transform from MatLike to NDArray
        modified_img = np.copy(image)
        # Define the region to modify (top-left coordinates and size)
        xc, yc = 64, 64
        xl, yl = 32, 96
        xr, yr = 96, 96
        size = 32

        # Change the brightness of the region (increase by 50)
        brightness_c = 50
        brightness_l = 100
        brightness_r = 150
        modified_image = self.modify_brightness(modified_img, xc, yc, size, brightness_c)
        modified_image = self.modify_brightness(modified_image, xl, yl, size, brightness_l)
        modified_image = self.modify_brightness(modified_image, xr, yr, size, brightness_r)
        # Convert the modified image to RGB format
        modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(modified_image_rgb)
        processed_image = ImageTk.PhotoImage(processed_image)
        self.processed_label.configure(image=processed_image)
        self.processed_label.image = processed_image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()