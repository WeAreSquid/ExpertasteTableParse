import cv2
import numpy as np
import re
from com.utils.act_with_image import ActWithImage 

class ImageHealthService(ActWithImage):
    def clean_image(self, img, filename, ocr_model):
        self.set_image(img)
        self.store_image(self.image, filename, prefix = '1_original')
        self.convert_image_to_grayscale()
        self.store_image(self.grayscale_image, filename, prefix = '2_grayscale')
        self.get_threshold_value(self.grayscale_image, ocr_model)
        self.threshold_image(self.grayscale_image)
        self.store_image(self.thresholded_image, filename, prefix = '3_threshold')
        self.invert_image(self.grayscale_image)
        self.store_image(self.inverted_image, filename, prefix = '4_inverted')
        self.dilate_image(self.inverted_image)
        self.store_image(self.dilated_image, filename, prefix = '5_dilated')
        self.find_contours(self.dilated_image)
        self.filter_contours_and_leave_only_rectangles(self.dilated_image)
        self.find_largest_contour_by_area(self.dilated_image)
        self.order_points_in_the_contour_with_max_area(self.dilated_image)
        self.calculate_new_width_and_height_of_image(self.dilated_image)
        self.apply_perspective_transform(self.image)
        self.store_image(self.perspective_corrected_image, filename, prefix = '6_perspective_corrected')
        return self.perspective_corrected_image       

    def get_threshold_value(self, img, ocr_model):
        # Get the dimensions of the image
        height, width = img.shape

        # Define the coordinates of the first quarter
        top_left_x = 0
        top_left_y = 0
        bottom_right_x = width // 4  # Half of the image width
        bottom_right_y = height // 4  # Half of the image height
        
        
        # Crop the first quarter of the image
        cropped_image = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        result = ocr_model.ocr(cropped_image, cls=True)
        card_out = []
        _sampler_list = []

        for line in result:
            for word_info in line:
                word = word_info[0]
                confidence = word_info[1]
                card_out.append((confidence[0], word))
                if re.search('sampler', confidence[0].lower()):
                    points = np.array(word)
                    centroid = np.mean(points, axis=0)
                    sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                    max_per_coordinate = np.max(sorted_points, axis=0)
                    min_per_coordinate = np.min(sorted_points, axis=0)
                    
                    weight = int(max_per_coordinate[0] - min_per_coordinate[0])
                    height = int(max_per_coordinate[1] - min_per_coordinate[1])
                    _sampler_list.append((confidence[0], word, weight, height, (centroid[0], centroid[1])))
            break

        # Crop the first quarter of the image
        for element in _sampler_list:
            self.pixel_color = img[int(element[4][1])+int(element[3]/2), int(element[4][0])]
            break
    
    def convert_contours_to_bounding_boxes(self, img):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = img.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            # This line below is about
            # drawing a rectangle on the image with the shape of
            # the bounding box. Its not needed for the OCR.
            # Its just added for debugging purposes.
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
            
    def process_perspective_image(self, img):
        result = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = img.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)
        self.image_with_only_rectangular_contours = img.copy()
        self.image_with_only_rectangular_contours_color = cv2.cvtColor(self.image_with_only_rectangular_contours, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.image_with_only_rectangular_contours_color, self.contours, -1, (0, 255, 0), 3)
        
    
    def convert_image_to_grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def blur_image(self):
        self.blurred_image = cv2.blur(self.grayscale_image, (5, 5))

    def threshold_image(self, img):
        self.thresholded_image = cv2.threshold(img, 255-self.pixel_color, 255, cv2.THRESH_BINARY)[1]       

    def close_image(self):
        kernel = np.ones((5,5), np.uint8)
        self.closed_image = cv2.bitwise_not(cv2.morphologyEx(self.dilated_image, cv2.MORPH_CLOSE, kernel))
        
    def invert_image(self, img):
        self.inverted_image = cv2.bitwise_not(img)

    def dilate_image(self, img):
        self.dilated_image = cv2.dilate(img, None, iterations=2)

    def find_contours(self, img):
        result = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = img.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, 0, (0, 255, 0), 3)

    def filter_contours_and_leave_only_rectangles(self, img):
        self.rectangular_contours = []
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                self.rectangular_contours.append(approx)
        self.image_with_only_rectangular_contours = img.copy()
        self.image_with_only_rectangular_contours_color = cv2.cvtColor(self.image_with_only_rectangular_contours, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.image_with_only_rectangular_contours_color, self.rectangular_contours, -1, (0, 255, 0), 3)
   
    def find_largest_contour_by_area(self, img):
        max_area = 0
        self.contour_with_max_area = None
        self.contours_list = []
        for contour in self.rectangular_contours:
            contour_data = {}
            contour_data['contour'] = contour
            area = cv2.contourArea(contour)
            contour_data['area'] = area
            if area > max_area:
                max_area = area
                self.contour_with_max_area = contour
            self.contours_list.append(contour_data)
        
        def extract_area(dict):
            return dict['area']
        
        self.contours_list.sort(key=extract_area, reverse=True)
        self.contour_with_max_area = self.contours_list[0]['contour']
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.image_with_contour_with_max_area = img.copy()
        
        cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)
        

    def order_points_in_the_contour_with_max_area(self, img):
        self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)
        self.image_with_points_plotted = img.copy()
        for point in self.contour_with_max_area_ordered:
            point_coordinates = (int(point[0]), int(point[1]))
            self.image_with_points_plotted = cv2.circle(self.image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)

    def calculate_new_width_and_height_of_image(self, img):
        existing_image_width = img.shape[1]
        existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        
        distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
        distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

        aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

        self.new_image_width = existing_image_width_reduced_by_10_percent
        self.new_image_height = int(self.new_image_width * aspect_ratio)

    def apply_perspective_transform(self, img):
        pts1 = np.float32(self.contour_with_max_area_ordered)
        pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        self.perspective_corrected_image = cv2.warpPerspective(img, matrix, (self.new_image_width, self.new_image_height))

    def add_10_percent_padding(self):
        image_height = self.image.shape[0]
        padding = int(image_height * 0.1)
        self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    def draw_contours(self):
        self.image_with_contours = self.image.copy()
        cv2.drawContours(self.image_with_contours,  [ self.contour_with_max_area ], -1, (0, 255, 0), 1)

    def calculateDistanceBetween2Points(self, p1, p2):
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis
    
    def order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect
    
    #### FURTHER CLEANING!!
    
    def erode_vertical_lines(self):
        self.new_inverted_image = cv2.bitwise_not(self.thresholded_image)
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.new_inverted_image, hor, iterations=5)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=10)

    def erode_horizontal_lines(self):
        ver = np.array([[1],
               [1],
               [1],
               [1],
               [1],
               [1],
               [1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.new_inverted_image, ver, iterations=10)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=10)

    def combine_eroded_images(self):
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)

    def dilate_combined_image_to_make_lines_thicker(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)

    def subtract_combined_and_dilated_image_from_original_image(self):
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)

    def remove_noise_with_erode_and_dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=1)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)
    