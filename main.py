import torch
from PIL import Image
from io import BytesIO
import requests
import flask
from flask import Flask

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.conf = 0.4

app = Flask(__name__)
@app.route("/")
def hello():
    return "Hello World!"
@app.route("/predict",methods=["POST"])
def predict():
    output_dict = {'success' : 'False'}
    if flask.request.method == "POST":
        data = flask.request.json
        imgs = []
        for img in data['image']:
            response = requests.get(img)
            imgs.append(Image.open(BytesIO(response.content)))
        results = model(imgs)
        results.print()
        output_dict['success'] = 'True'
        output_dict['predict'] = []
        for p in results.pandas().xyxy:
            output_dict['predict'].append(p.to_dict('records'))
    return output_dict,200
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)