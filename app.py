from distutils.log import debug
from flask import Flask 
import os
import psycopg2
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, make_response


labels = ["ayam dada", "ayam paha", "bakso", "burger", "gado-gado", "kebab", "kentang", "makaroni", "rendang", "roti", "sosis", "takoyaki"]

def get_db():
    conn = psycopg2.connect(
        host="",
        database="",
        user=os.environ['DB_USERNAME'],
        password=os.environ['DB_PASSWORD'])
    return conn

def predict_image(IMG_PATH):
    model = load_model("")

    img = image.load_img(IMG_PATH, target_size=(500, 375, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    
    results = classes.flatten()

    for (label, result) in zip(labels, results):
        if (result == 1):
            return label
    

app = Flask(__name__)
cors = CORS(app)

@app.route('/predict', methods=["POST"])
def predict_process():
    img = request.files["img"]

    img_file = str(uuid.uuid1()) + ".jpg"
    img.save(img_file)

    predict_result = predict_image(img_file)

    conn = get_db()
    cur = conn.cursor()
    result = cur.execute("select * from foods where name like '$1'", [predict_result])
    
    return make_response(jsonify(result))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
