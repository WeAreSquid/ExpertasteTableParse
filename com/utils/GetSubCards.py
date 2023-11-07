from service.image_health_service import ImageHealthService

class GetSubCards(ImageHealthService):
    def __init__(self):
        image_path = r"./images/12_perspective_corrected_with_padding.jpg"
        TableExtractor().__init__()
    
    def get_sub_cards(self):
        print(self.inverted_image)

if __name__ == "__main__":
    """
    image_path = r"./images/12_perspective_corrected_with_padding.jpg"
    image_to_parse = TableExtractor(image_path, suffix = "_corrected" )
    perspective_corrected_image = image_to_parse.main_execution()
    """
    get_cards = GetSubCards()
    get_cards.get_sub_cards()
       