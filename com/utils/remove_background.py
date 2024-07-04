from PIL import Image
import numpy as np
import cv2
import io
from rembg import remove

class RemoveBackground:
    def __init__(self, img):
        self.img = img

    def main(self):
        image_no_bg = self.remove_background(self.img)
        return image_no_bg

    def remove_background(self, img, border=0):
        image = Image.fromarray(img)
        output = remove(image) 
        output_array = np.array(output)
        foreground_coords = np.argwhere(output_array[:, :, 3] > 0)
        (min_y, min_x), (max_y, max_x) = foreground_coords.min(0), foreground_coords.max(0)

        cropped_image = output_array[min_y:max_y, min_x:max_x]
        cropped_image_pil = Image.fromarray(cropped_image)
        background_color=(0, 0, 0)
 
        if cropped_image_pil.mode in ('RGBA', 'LA') or (cropped_image_pil.mode == 'P' and 'transparency' in cropped_image_pil.info):
            background = Image.new('RGB', cropped_image_pil.size, background_color)
            background.paste(cropped_image_pil, mask=cropped_image_pil.split()[3])
            img = background.convert('RGB')
        else:
            img = cropped_image_pil.convert('RGB')
        img_np = np.array(img)
        return img_np