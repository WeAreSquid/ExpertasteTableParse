import cv2
import easyocr
import numpy as np
import subprocess
import pytesseract
from PIL import Image
from com.utils.act_with_image import ActWithImage
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image


class OCRService(ActWithImage):
    def execute_easy(self, img):
        self.set_image(img)
        self.original_image = img
        #image = cv2.imread(self.image)
        render = easyocr.Reader(['en'],download_enabled=False, model_storage_directory=r'C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\com\service\english_g2')
        
        results = render.readtext(img)
        
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            print(bbox, text, prob)
            
        return
    def execute_paddleocr(self, img):
        self.set_image(img)
        self.original_image = img

        # Initialize PaddleOCR
        ocr = PaddleOCR(lang='en')
        ocr.det_algorithm = 'DB-RESNET'

        # Perform OCR
        result = ocr.ocr(self.image, cls=True)
        
        # Process OCR results
        for line in result:
            for word_info in line:
                word = word_info[0]
                confidence = word_info[1]
                print(f'Word: {word}, Confidence: {confidence}')

        # Visualization (optional)
        image = Image.open(self.image).convert('RGB')
        boxes = [elements[0] for line in result for elements in line]
        pairs = [elements[1] for line in result for elements in line]
        txts = [pair[0] for pair in pairs]
        scores = [pair[1] for pair in pairs]

        im_show = draw_ocr(image, boxes, txts, scores)

        # Display the image with bounding boxes and recognized text
        plt.imshow(im_show)
        plt.axis('off')
        plt.show()
            
        return
        

class OCR2Service(ActWithImage):
    def execute(self, img):
        self.set_image(img)
        self.original_image = img

        #self.dilate_image()
        #self.store_process_image('testing_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('testing_dilated_image_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        print('1')
        self.store_process_image('testing_dilated_image_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        print('2')
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        print('3')
        self.sort_bounding_boxes_by_y_coordinate()
        print('4')
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        print('5')
        self.sort_all_rows_by_x_coordinate()
        print('6')
        self.crop_each_bounding_box_and_ocr()
        print('7')
        self.generate_csv_file()
        print('8')

    def threshold_image(self):
        return cv2.threshold(self.grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def convert_image_to_grayscale(self):
        return cv2.cvtColor(self.image, self.dilated_image)

    def dilate_image(self):
        kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1]
        ])
        self.dilated_image = cv2.dilate(self.image, kernel_to_remove_gaps_between_words, iterations=5)
        simple_kernel = np.ones((5,5), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=2)
    
    def find_contours(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        result = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
    
    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 5, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)

    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [ bounding_box ]
        self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])

    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        current_row = []
        image_number = 0
        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = y - 5
                cropped_image = self.original_image[y:y+h, x:x+w]
                image_slice_path = "./testing_" + str(image_number) + ".jpg"
                cv2.imwrite(image_slice_path, cropped_image)
                results_from_ocr = self.get_result_from_tersseract(image_slice_path)
                current_row.append(results_from_ocr)
                image_number += 1
            self.table.append(current_row)
            current_row = []

    def get_result_from_tersseract(self, image_path):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        output = pytesseract.image_to_string(Image.open(image_path), lang = 'eng', config='--oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "')
        output = output.strip()
        return output

    def generate_csv_file(self):
        with open(r"C:\Users\ferbo\Desktop\GoogleDrive_ferboubeta2\other_projects\TableInImage\ExpertasteTableParse\output.csv", "w") as f:
            for row in self.table:
                print(row)
                f.write(",".join(row) + "\n")

    def store_process_image(self, file_name, image):
        path = "./" + file_name
        cv2.imwrite(path, image) 