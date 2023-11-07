from com.utils.work_points import workPoints

class GetSubCards():    
    def get_cards(self, samplers_points, image):
        work_with_points = workPoints()
        image_dictionary = []
        for element in samplers_points['entities']:
            _sampler_points = work_with_points.order_points(element)
            order_crop_element = work_with_points.crop_cards(_sampler_points, image)
            image_dictionary.append(order_crop_element)
        results_output = {'samplers_results': image_dictionary}
        return results_output