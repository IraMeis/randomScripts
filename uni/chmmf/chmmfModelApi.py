"""
Run a Flask REST API
"""

import argparse
import io
from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from uni.chmmf.chmmfModel import ChmmfModel

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def parse_request_for_model(data: dict):
    R = float(data["RFiz"])
    T = float(data["TFiz"])
    k = float(data["kFiz"])
    c = float(data["cFiz"])
    aI = int(data["I"])
    aK = int(data["K"])
    return ChmmfModel(R=R, T=T, k=k, c=c, amount_I=aI, amount_K=aK)


@app.route('/api/chmmf/t', methods=["POST"])
@cross_origin()
def model_by_t():
    data = request.form.to_dict()
    isx = bool(int(data["isxlim"]))
    isy = bool(int(data["isylim"]))

    model = parse_request_for_model(data)
    matrix = model.schema_solution()

    if isy and isx:
        plt = model.showPlotsByT(matrix, [10],
                                 ylim=tuple(map(float, data["ylim"].split(','))),
                                 xlim=tuple(map(float, data["xlim"].split(','))))
    elif isx:
        plt = model.showPlotsByT(matrix, [10], xlim=tuple(map(float, data["xlim"].split(','))))
    elif isy:
        plt = model.showPlotsByT(matrix, [10], ylim=tuple(map(float, data["ylim"].split(','))))
    else:
        plt = model.showPlotsByT(matrix, [10])

    img = io.BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/api/chmmf/r', methods=["POST"])
@cross_origin()
def model_by_r():
    data = request.form.to_dict()
    isx = bool(int(data["isxlim"]))
    isy = bool(int(data["isylim"]))

    model = parse_request_for_model(data)
    matrix = model.schema_solution()

    if isy and isx:
        plt = model.showPlotsByR(matrix, [10],
                                 ylim=tuple(map(float, data["ylim"].split(','))),
                                 xlim=tuple(map(float, data["xlim"].split(','))))
    elif isx:
        plt = model.showPlotsByR(matrix, [10], xlim=tuple(map(float, data["xlim"].split(','))))
    elif isy:
        plt = model.showPlotsByR(matrix, [10], ylim=tuple(map(float, data["ylim"].split(','))))
    else:
        plt = model.showPlotsByR(matrix, [10])

    img = io.BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API for model")
    parser.add_argument("--port", default=5088, type=int, help="port number")
    opt = parser.parse_args()
    app.run(host="0.0.0.0", port=opt.port)
