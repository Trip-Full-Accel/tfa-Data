import os
import shutil
import urllib
import cv2
import flask
import pymysql
import numpy as np
from flask import jsonify
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import keras.utils as image
from keras.models import load_model

app = flask.Flask(__name__)


def get_image():
    conn = pymysql.connect(host='localhost',
                           user='root',
                           password='1234',
                           db='tfa',
                           charset='utf8')
    curs = conn.cursor()
    sql = "SELECT id, url FROM post ORDER BY created_at DESC LIMIT 20"
    curs.execute(sql)

    images = curs.fetchall()
    images_list = []
    for obj in images:
        images_dict = {
            "id": obj[0],
            "image": obj[1]
        }
        images_list.append(images_dict)

    curs.close()
    conn.close()

    return images_list


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    print(resp)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    resp.close()

    return image


@app.route("/recommend/image", methods=["POST"])
def recommend_image():
    if os.path.exists("./temp/image/"):
        shutil.rmtree("./temp/image/")
    os.mkdir("./temp/image/")

    images_list = get_image()
    test_images = []
    check = 0
    for i in images_list:
        check += 1
        i["image"] = urllib.parse.quote_plus(i["image"], safe="://?=&")
        test_image = url_to_image(i["image"])

        urllib.request.urlretrieve(i["image"], "./temp/image/" + str(check) + ".jpg")
        test_images.append(test_image)

    day_images = []
    night_images = []

    model = load_model("./day_night_model.h5")

    test_dir = './temp/image'
    test_filenames = os.listdir(test_dir)

    dic_ox_filenames = {}
    dic_ox_filenames["image"] = test_filenames

    for ox, filenames in dic_ox_filenames.items():

        fig = plt.figure(figsize=(16, 10))
        rows, cols = 1, 20
        for i, fn in enumerate(filenames):
            path = test_dir + '/' + fn
            test_img = image.load_img(path, target_size=(150, 150), interpolation='bilinear')
            x = image.img_to_array(test_img)
            x = np.expand_dims(x, axis=0)
            temp = np.vstack([x])

            classes = model.predict(temp, batch_size=10)

            fig.add_subplot(rows, cols, i + 1)

            if classes[0] == 0:
                plt.title(" is O")
                plt.axis('off')
                plt.imshow(test_img, cmap='gray')

                day_images.append(images_list[i])

            else:
                plt.title(" is X")
                plt.axis('off')
                plt.imshow(test_img, cmap='gray')

                night_images.append(images_list[i])
        plt.show()

    return jsonify(day_images, night_images)
