import cv2
from PIL import Image
from flask import Flask, jsonify, request
import numpy as np

from yolov4_tiny import YOLOv4Tiny


app = Flask(__name__)
model = None


def load_model() -> None:
    global model
    try:
        model = YOLOv4Tiny(device="cpu")
    except Exception as e:
        print("Failed to load the model. Error:", e)
        raise e
    print("Model initialized")


def create_app() -> Flask:
    load_model()

    return app


@app.route("/api/v1/towers", methods=["POST"])
def towers():
    if "image" not in request.files:
        app.logger.debug("No file in request")
        return jsonify(
            {"status": "error", "msg": "no image in request"}
        ), 400

    response = {
        "status": "success",
        "image size": None,
        "prediction": None
    }

    try:
        image_file = request.files["image"]
        image = cv2.cvtColor(
            np.array(Image.open(image_file.stream)),
            cv2.COLOR_RGB2BGR
        )
        preds = model.predict([image])[0]  # batch 1
        preds = [
            (pred[-1], pred[-2]) for pred in preds if pred
        ]
        response["image_size"] = image.shape
        response["prediction"] = " ".join([str(e) for e in preds])
    except Exception as e:
        app.logger.debug("Error happened:", e)
        response["status"] = "failed"

    return jsonify(response)


if __name__ == "__main__":
    load_model()
    app.run(threaded=False)
