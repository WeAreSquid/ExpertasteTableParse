import numpy as np
from com.utils.act_with_image import ActWithImage
from com.utils.get_sub_cards import GetSubCards
import re


class OCRCheck(ActWithImage):
    def __init__(self):
        self.output_dict = {}
        
    def main_execution(self, img, ocr_model):
        self.ocr_output = self.execute_paddleocr(img, ocr_model)
        get_sub_cards = GetSubCards()
        self.card_info_dict = get_sub_cards.get_cards(self.ocr_output, img)
        return self.card_info_dict
        
    def execute_paddleocr(self, img, ocr_model):
        # Get the height and width of the image
        height, width, channels = img.shape
         
        top_left_x = 0
        top_left_y = 0

        bottom_right_x = int(width / 3)
        bottom_right_y = height
        cropped_image = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        result = ocr_model.ocr(cropped_image, cls=True)
        card_out = []
        _sampler_list = []

        _first_entities_list = []
        for line in result:
            for word_info in line:
                word = word_info[0]
                confidence = word_info[1]
                card_out.append((confidence[0], word))
                _sampler_dict = {}
                points = np.array(word)
                centroid = np.mean(points, axis=0)
                sorted_points = sorted(points, key=lambda point: (-np.arctan2(point[1] - centroid[1], point[0] - centroid[0])))
                max_per_coordinate = np.max(sorted_points, axis=0)
                min_per_coordinate = np.min(sorted_points, axis=0)
                
                width = int(max_per_coordinate[0] - min_per_coordinate[0])
                height = int(max_per_coordinate[1] - min_per_coordinate[1])
                _sampler_dict['name'] = confidence[0].lower()
                _sampler_dict['coordinates'] = word
                _sampler_dict['width'] = width
                _sampler_dict['height'] = height
                _sampler_dict['centroid'] = [centroid[0], centroid[1]]
                if re.search('sampler', confidence[0].lower()):
                    _sampler_list.append(_sampler_dict)
                else:
                    _first_entities_list.append(_sampler_dict)
                    
        self.sampler_list = _sampler_list
        self.first_entities_list = _first_entities_list
        
        if len(_sampler_list) != 0:
            self.output_dict['status'] = 'pass'
            self.output_dict['entities'] = _sampler_list 
        else:
            self.output_dict['status'] = 'failed'
            self.output_dict['entities'] = '[]'    
        return self.output_dict
    
    def group_entities(self):
        self.output_dict['first_entities_found'] = []
        for sampler in self.sampler_list:
            _name_sampler = sampler['name']
            _width_sampler = sampler['width']
            _height_sampler = sampler['height']
            _centroid_sampler = sampler['centroid']
            entitites_list = []
            for entity in self.first_entities_list:
                _name_entity = entity['name']
                _width_entity = entity['width']
                _height_entity = entity['height']
                _centroid_entity = entity['centroid']
                if _centroid_entity[1] >= _centroid_sampler[1] - int(_width_sampler/2.5) and _centroid_entity[1] <= _centroid_sampler[1] + int(_width_sampler/2.5):
                    entitites_list.append({'name': _name_entity, 'width': _width_entity, 'height': _height_entity, 'centroid' : _centroid_entity})
            self.output_dict['first_entities_found'].append({'name': _name_sampler, 'first_entities': entitites_list})
                       
        return self.output_dict
                