import json, requests

class RequestService():
    def make_request(self, URL, json_body):
        header = {}
        header['Content-Type'] = 'application/json'
        header = json.loads(str(header))
        try:
            response = requests.request('POST', headers = header, url = URL, json = json_body)
            return response
        except Exception as e:
            return {'message': e}
            
    
        