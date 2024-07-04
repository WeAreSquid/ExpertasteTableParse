import numpy as np
import cv2
from PIL import Image
import imutils

class PerspectiveCorrection:
    def perspective_correction(self, img, padding_size = 100, padding_color=(0, 0, 0)):
        self.padded_image = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=padding_color)

        """Find the largest square contour in the image."""
        gray = cv2.cvtColor(self.padded_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 100, cv2.THRESH_BINARY)
        
        cnts = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

        # loop over the contours
        rectangular_contours = []
        max_area = 0
        contour_with_max_area = None

        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                area = cv2.contourArea(screenCnt)
                if area > max_area:
                    max_area = area
                    contour_with_max_area = screenCnt

        # show the contour (outline) of the piece of paper
        cv2.drawContours(self.padded_image, [contour_with_max_area], -1, (0, 255, 0), 10)
  
        
        if contour_with_max_area is not None:
            # Crop the image to the largest square contour
            x, y, w, h = cv2.boundingRect(contour_with_max_area)
            cropped_image = self.padded_image[y:y+h, x:x+w]

            # Convert to RGB (if necessary) and save the result
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB) 
            cropped_image_pil = Image.fromarray(cropped_image_rgb)
            cropped_image_pil = np.array(cropped_image_pil)


        pts = contour_with_max_area.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        """Apply perspective correction to the image given the source points."""
        width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))

        dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        result = cv2.warpPerspective(img, matrix, (int(width), int(height)))
        return result