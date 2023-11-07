from fastapi import APIRouter, File, UploadFile
import json
import urllib.parse
import cv2
import numpy as np
from paddleocr import PaddleOCR

from com.models.input_model import InputModel
from com.service.image_health_service import ImageHealthService

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
            return json.loads(json_string)
    except Exception as e:
        print(e)
        return {'message': e}
        
    """
    try:
        #Calling next service
        URL = 'url for next service'
        header = {'content-type':'application/json'}
        header = json.loads(header)
        async with httpx.AsyncClient() as client:
            response = await client.request('POST', URL, json = data_form, header = header, follow_redirects = True, timeout = None)
            return {'message': 'Service completed', 'status code': response.status_code}
    except httpx.HTTPStatusError as exc:
        print('Failed!')
    """
        
        
    
    