"""
Run a Flask REST API
"""

import argparse
import threading
import time
import numpy as np
import io
from flask import Flask, request, send_file
from PIL import Image
from flask_cors import CORS, cross_origin
from flask_api import status
from uni.matmod.model import Model


def refresh_image():
    global model
    if model.is_running:
        model.calculate_f_nexts_update_f_renders()
    else:
        time.sleep(10)
    refresh_image()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = None
th = threading.Thread(target=refresh_image, daemon=True)


@app.route('/api/matmod/new', methods=["POST"])
@cross_origin()
def create_model():
    global model
    data = request.form.to_dict()
    N = np.int_(data["N"])
    As = np.array(list(map(float, data["As"].split(','))), dtype=np.float_)
    Ds = np.array(list(map(float, data["Ds"].split(','))), dtype=np.float_)
    matrix = np.reshape(np.array(list(map(float, data["matrix"].split(','))), dtype=np.float_), (N, N))
    It = float(data["It"])
    Iy = float(data["Iy"])
    Ix = float(data["Ix"])
    model = Model(N=N, As=As, Ds=Ds, matrix=matrix, It=It, Iy=Iy, Ix=Ix)
    imgs = []
    for i in range(int((N - 1) / 3) + 1):
        try:
            file = request.files[str(i)]
            imgs.append(Image.open(io.BytesIO(file.read())))
        except Exception as e:
            print(e)
            return "BAD REQUEST", status.HTTP_400_BAD_REQUEST
    model.set_f_nexts0_set_f_renders0(imgs)
    for i in range(int((N - 1) / 3) + 1):
        imgs[i].close()
    model.is_running = True
    th.start()
    return "", status.HTTP_200_OK


@app.route('/api/matmod/update/<numb>', methods=["GET"])
@cross_origin()
def send_img(numb):
    img_io = io.BytesIO()
    img = Image.fromarray(model.f_renders[int(numb)])
    sw, sh = img.size
    img = img.resize((int(sw * 4), int(sh * 4)))
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API for model")
    parser.add_argument("--port", default=5099, type=int, help="port number")
    opt = parser.parse_args()
    app.run(host="0.0.0.0", port=opt.port)
