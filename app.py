from flask import Flask, render_template, request, url_for, redirect
# import tensorflow as tf
# import numpy as np
# from PIL import Image
from werkzeug.utils import secure_filename
import os
'''
sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

tf.compat.v1.keras.backend.set_session(sess)
digits_rec_model = tf.keras.models.load_model('digits_rec_model.h5')'''

port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/contact_me')
def contact_me():
    return render_template('contact.html')


@app.route('/digits_recognition', methods=['POST', 'GET'])
def digits_recognition():
    if request.method == 'POST' and request.files:
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('digits_recognition_answer', filename=filename))
    return render_template('digits_recognition.html')


@app.route('/digits_recognition/<filename>', methods=['POST', 'GET'])
def digits_recognition_answer(filename):
    '''  
    im = Image.open(os.path.join('uploads', filename))
    im = im.convert('L').resize((28, 28))
    im = np.array(im, dtype='int32')
    im = np.reshape(im, (1, 784))

    with graph.as_default():
        tf.compat.v1.keras.backend.set_session(sess)
        prediction = digits_rec_model.predict_classes(im)[0]
    '''
    prediction = 100
    return render_template('digits_recognition-answer.html', answer=prediction)


if __name__ == '__main__':
    app.run(debug = True, threaded=False, host='0.0.0.0', port=port)
