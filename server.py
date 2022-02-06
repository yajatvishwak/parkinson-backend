
import json
from flask import Flask ,  request , jsonify
import tensorflow as tf
from flask_cors import CORS


import os


app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



import numpy as np
from tensorflow import keras
class_names = ["no park", "park"]


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict" , methods=["POST"])
def predict():
    f = request.files['file']
    f.save(os.path.join(UPLOAD_FOLDER, f.filename))
    
    img = tf.keras.preprocessing.image.load_img(
        os.path.join(UPLOAD_FOLDER, f.filename), target_size=(300, 300)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(class_names[np.argmax(score)])
    return jsonify({"pred":class_names[np.argmax(score)]})
import base64
from PIL import Image
import numpy as np

@app.route("/predict2" , methods=["POST"])
def predict2():
    request_data = request.get_json()

    img = request_data['img']
    img = img[img.find(",")+1:]
    print(img)
    # with open(os.path.join(UPLOAD_FOLDER, "imageToSave.png"), "wb") as fh:
    #     fh.write(base64.decodebytes(img))
    with open(os.path.join(UPLOAD_FOLDER, "imageToSave.png"), "wb") as fh:
        fh.write(base64.urlsafe_b64decode(str(img)))
    im = Image.open(os.path.join(UPLOAD_FOLDER, "imageToSave.png"))
    n = np.array(im)
    n[...,0:3]=[0,0,0]
    Image.fromarray(n).save(os.path.join(UPLOAD_FOLDER, "imageToSave.png"))

    fill_color = (255,255,255)
    im = im.convert("RGBA")
    if im.mode in ('RGBA', 'LA'):
        background = Image.new(im.mode[:-1], im.size, fill_color)
        background.paste(im, im.split()[-1]) # omit transparency
        im = background
    im.convert("RGB").save(os.path.join(UPLOAD_FOLDER, "imageToSave.png"))

    
    img = tf.keras.preprocessing.image.load_img(
        os.path.join(UPLOAD_FOLDER, "imageToSave.png"), target_size=(300, 300)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(class_names[np.argmax(score)])
    return jsonify({"pred":class_names[np.argmax(score)]})

if __name__ == '__main__':
    model = keras.models.load_model('./model/neww.h5')
    app.run( debug=True,port=5000)