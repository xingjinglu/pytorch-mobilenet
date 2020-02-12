import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
import os

def tv_transform():



def read_image():
    #img = Image.open(os.path.join("ILSVRC2012_val_00000396", ".JPEG"))
    img = Image.open(os.path.join("ILSVRC2012_val_00000396.JPEG"))
    #img.show()
    print(img.size) # width, height
    print(img.mode)

    img = np.array(img, dtype = np.uint8)   
    print(img.size)
   
    img = Image.fromarray(img.astype(np.uint8))  # h, w, c
    img.save("images_13.jpg")
    img_resized = img.resize((244, 244))
                     


def main():
    read_image()
    tv_transform()

if __name__ == "__main__":
    main()
