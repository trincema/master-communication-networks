import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# 3.b, 3.c is valid only for C++, Python allocates memory automatically
class ImageProcessorApp:
    def __init__(self, root):
        self.N = 32     # 2.a, 3.a

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
    
    def show_grayscale_image(self):
        if self.image_path:
            # Example: Process image using OpenCV
            image = cv2.imread(self.image_path)
            modified_img = np.copy(image)
            # Get the dimensions of the image
            height, width, channels = image.shape
            print("Image dimensions - Height:", height, "Width:", width, "Channels:", channels)
            # Calculate how many square blocks will fit horizontally
            blocks = 0
            if width >= height: # Make sure we don't try to iterate outside the image
                blocks = int(width / self.N)
            else:
                blocks = int(height / self.N)
            print("Square blocks fit horizontally/vertically: " + str(blocks))
            modified_image = None
            # 5 - Loop in blocks and change brightness
            for row in range(0, blocks):
                for column in range(0, blocks):
                    x = row * self.N
                    y = column * self.N
                    brightness = 0
                    if row % 2 == 0:
                        if column % 2 == 0:
                            brightness = 75
                        else:
                            brightness = 25
                    else:
                        if column % 2 == 0:
                            brightness = 25
                        else:
                            brightness = 75
                    modified_image = self.modify_brightness(modified_img, x, y, self.N, brightness)

            # Perform image processing here
            # 3.d, 4 Display processed grayscale image, store it in a new variable
            # 4 The processed image will be derived from original image, so no duplication needed in Python
            processed_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)  # Example: Convert to grayscale
            processed_image = Image.fromarray(processed_image)
            processed_image = ImageTk.PhotoImage(processed_image)
            self.processed_label.configure(image=processed_image)
            self.processed_label.image = processed_image
    
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
        # 5 - Extract the region of interest
        roi = img[y:y+size, x:x+size]
        # 6 - Change the brightness of the region
        roi = cv2.add(roi, brightness_change)
        # Clip the pixel values to the valid range [0, 255]
        roi = np.clip(roi, 0, 255)
        # Replace the region in the modified image
        img[y:y+size, x:x+size] = roi
        return img
    
    def process_image(self):
        self.show_grayscale_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()