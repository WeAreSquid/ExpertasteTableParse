from fastapi import APIRouter, File, UploadFile
import json
import requests
import urllib.parse
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from paddleocr import PaddleOCR

from com.models.input_model import InputModel
from com.service.image_health_service import ImageHealthService
from com.service.request_service import RequestService
request_service =  RequestService()

from com.utils.ocr_check import OCRCheck

router = APIRouter()
@router.post('/image_health')
async def image_health_rest(file: UploadFile = File(...)):
    file_bytes = await file.read()
    filename = urllib.parse.unquote_plus(file.filename)
    np_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    try:
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_health = ImageHealthService()
            ocr_model = PaddleOCR(ocr_version="PP-OCRv3",lang='en', rec_algorithm = 'CRNN', rec_char_type = 'en', use_angle_cls=True)
            image_cleaned = image_health.clean_image(img, filename, ocr_model)
            ocr_check = OCRCheck()
            results = ocr_check.main_execution(image_cleaned, ocr_model)
            json_string = json.dumps(results)
            json_data = json.loads(json_string)
    except Exception as e:
        return {'message': e}
    
    session = requests.Session()
    headers = {'Content-Type': 'multipart/form-data'}
    url = 'PUT URL HERE!'
    for element in json_data['samplers_results']:
        try:
            # Changing from Base64 to BytesIO to be send
            decoded_bytes = base64.b64decode(element['cropped_card'])
            array = np.frombuffer(decoded_bytes, dtype=np.uint8)
            original_shape = element['array_shape']
            image_array = array.reshape((original_shape[0], original_shape[1], original_shape[2]))
            image = Image.fromarray(image_array)
            image_buffer = BytesIO()
            image.save(image_buffer, format='PNG')
            files = {'file': ('image.png', image_buffer.getvalue(), 'image/png')}
            #response = session.post(url, files=files, headers=headers)
            ind = json_data['samplers_results'].index(element)
            json_data['samplers_results'][ind]['cropped_card'] = 'RESPONSE URL HERE! for ' + element['name']
            del json_data['samplers_results'][ind]['array_shape']
            
        except Exception as e:
            return {'message': e}
    
    return json_data
    