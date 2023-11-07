from fastapi import FastAPI
from com.rest import image_health_rest
import uvicorn

app = FastAPI()
app.include_router(image_health_rest.router)

def run_server():
    print('Running!')
    uvicorn.run("main:app", port = 4100, host = '0.0.0.0', reload = True)
    
if __name__ == '__main__':
    run_server()