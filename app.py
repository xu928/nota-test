from flask import Flask, jsonify, request, render_template
from train import get_prediction
app = Flask(__name__)

@app.route('/test')
def test():
    return "fork test3"

@app.route('/train')
def train():
    from train import mnist_train
    e = request.args.get('epoch', default=1)
    training_epochs = int(e)
    batch_size = 32
    mnist_train(training_epochs, batch_size)

    return jsonify({'train_complete': True,
                    'train_epochs': training_epochs,
                    "batch_size": batch_size})

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


app.run(host="0.0.0.0")