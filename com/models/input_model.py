from pydantic import BaseModel
from fastapi import FastAPI, UploadFile 

app = FastAPI()

class InputModel(BaseModel):
    file: UploadFile


