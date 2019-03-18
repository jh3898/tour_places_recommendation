from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from flask import Flask, render_template, request
from os import listdir, walk
from os.path import isfile, join
import numpy as np
import json

def getAllFilesInDirectory(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f)) and f[-4:] == '.jpg']


def read_data():
    trained_dic = 'holiday_photos_trained_dic.json'
    with open(trained_dic) as infile:
        data = json.load(infile)
    return data
dat = read_data()

def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

def findDifference(f1, f2):
    return np.linalg.norm([a - b for a, b in zip(f1, f2)])

def driver(IMAGE_DIR):
    feature_vectors: dict = {}
    model = ResNet50(weights='imagenet')
    for img_path in getAllFilesInDirectory(IMAGE_DIR):
        feature_vectors[img_path] = predict(img_path,model)[0].tolist()
    return feature_vectors


def predict_all(image_feature_vectors, custom_feature_vectors):
    k = list(custom_feature_vectors.keys())[0]
    diff = {}
    for v in image_feature_vectors:
        diff[v] = findDifference(custom_feature_vectors[k], image_feature_vectors[v])
        # if diff < ans:
        # similar[k] = v
    A = sorted(diff.items(), key=lambda x: x[1])[:3]
    print(A)
    similar = [item[0].split('/')[-1] for item in A]
    print(similar)
    return similar
#########################################app######################################
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/input')
def input_criteria():
    return render_template('input.html')


@app.route('/output', methods=['GET', 'POST'])
def output_criteria():
    f = ''
    if request.method == 'POST':
        f = request.files['file']
        #print(app.config['uploaded_image'], join(app.config['uploaded_image'], f.filename))
        f.save( f.filename)
        #return 'file upload successfully'
    print(f, f.filename)
    custom_im = driver("./")
    recommendation_out = predict_all(dat, custom_im)
    print(recommendation_out[0])
    #print('city, price_range:', city, price_range, restaurants_out['Name'].values, restaurants_out['Cuisine_Style'].values )
    return render_template('output.html', tour_name = recommendation_out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    #res = recommendation_output("Amsterdam", 1)
    #print(res['Name'].values, res['Cuisine_Style'].values)