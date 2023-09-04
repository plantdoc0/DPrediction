from flask import Flask, request, jsonify,send_file
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, firestore
from werkzeug.utils import secure_filename
import os
import base64
from PIL import Image
from flask import render_template_string

cred = credentials.Certificate('rubber-test-2f1f0-firebase-adminsdk-bn1h9-689ef8a3d4.json')
firebase_admin.initialize_app(cred)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'images'

db = firestore.client()

app = Flask(__name__)

def load_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        return str(e)

def adjust_brightness_and_saturation(image):
    image = tf.image.adjust_brightness(image, delta=0.2)
    image = tf.image.adjust_saturation(image, saturation_factor=1.2)
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_and_reduce_quality(image_file):
    """Resize the image and reduce its quality."""
    # Open the image using PIL
    img = Image.open(image_file)
    # Convert the image to JPEG and reduce its quality to, say, 75%
    output = BytesIO()
    img.save(output, format='JPEG', quality=75)
    output.seek(0)
    
    return output

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'API is working!'}), 200

@app.route('/leaf_predict', methods=['POST'])
def predict():
    interpreter = load_model('2_best_leaf_model.tflite')
    if isinstance(interpreter, str):
        return jsonify({'error': interpreter}), 500
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess image
    image = Image.open(request.files['image'].stream).resize((250, 250))
    image_arr = np.asarray(image) / 255.0  # Convert to numpy array and normalize
    image_arr = adjust_brightness_and_saturation(image_arr)
    image_arr = np.expand_dims(image_arr, axis=0)  # Expand dimensions for batch input
    image_arr = image_arr.astype(np.float32)  # Cast to float32
    
    class_labels = ['disease1','disease2','disease3','healthy']

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], image_arr)
    
    # Invoke the interpreter to perform inference
    interpreter.invoke()

    # Retrieve the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    interpreter = None  # Clear the variable

    return jsonify({'class': str(class_labels[predicted_class]), 'confidence': str(confidence)})

@app.route('/sheet_predict', methods=['POST'])
def sheet_predict():
    interpreter = load_model('best_sheet.tflite')
    if isinstance(interpreter, str):
        return jsonify({'error': interpreter}), 500
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess image
    image = Image.open(request.files['image'].stream).resize((150, 150))
    image_arr = np.asarray(image) / 255.0  # Convert to numpy array and normalize
    image_arr = adjust_brightness_and_saturation(image_arr)
    image_arr = np.expand_dims(image_arr, axis=0)  # Expand dimensions for batch input
    image_arr = image_arr.astype(np.float32)  # Cast to float32
    
    sheet_class_labels = ['RSS2','RSS3','RSS4 or RSS5' ,'Sole Crepe', 'Thick Crepe']  # Adjust this list accordingly based on your model's classes

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], image_arr)
    
    # Invoke the interpreter to perform inference
    interpreter.invoke()

    # Retrieve the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    interpreter = None  # Clear the variable

    return jsonify({'class': str(sheet_class_labels[predicted_class]), 'confidence': str(confidence)})

@app.route('/data', methods=['POST'])
def create_data():
    if 'image1' not in request.files or 'image2' not in request.files or 'image3' not in request.files:
        return jsonify({'error': 'Image files not provided'}), 400

    data_type = request.form['type']
    data_class = request.form['class']

    # Create directory structure if not exists
    save_path = os.path.join(UPLOAD_FOLDER, data_type, data_class)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save images and get their paths
    image1_filename = secure_filename(request.files['image1'].filename)
    request.files['image1'].save(os.path.join(save_path, image1_filename))

    image2_filename = secure_filename(request.files['image2'].filename)
    request.files['image2'].save(os.path.join(save_path, image2_filename))

    image3_filename = secure_filename(request.files['image3'].filename)
    request.files['image3'].save(os.path.join(save_path, image3_filename))

    data = {
        'id': request.form['id'],
        'type': data_type,
        'class': data_class,
        'topic1': request.form['topic1'],
        'description1': request.form['description1'],
        'topic2': request.form['topic2'],
        'description2': request.form['description2'],
        'image1_path': os.path.join(save_path, image1_filename),
        'image2_path': os.path.join(save_path, image2_filename),
        'image3_path': os.path.join(save_path, image3_filename),
    }

    db.collection('data').add(data)
    return jsonify({'message': 'Data added successfully'}), 201

@app.route('/data/images/<data_id>', methods=['GET'])
def get_images_by_data_id(data_id):
    # Query Firestore based on the 'id' field in the data
    docs = db.collection('data').where('id', '==', data_id).stream()

    for doc in docs:  # This loop should only run once since 'id' should be unique
        data = doc.to_dict()
        image1_path = data['image1_path']
        image2_path = data['image2_path']
        image3_path = data['image3_path']

        # Read the images and convert to base64 for embedding in HTML
        with open(image1_path, "rb") as img_file:
            image1_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        with open(image2_path, "rb") as img_file:
            image2_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        with open(image3_path, "rb") as img_file:
            image3_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Create a simple HTML page to display the images
        template = """
        <html>
            <body>
                <h2>Images for ID: {{id}}</h2>
                <img src="data:image/jpeg;base64,{{image1}}" width="300px">
                <img src="data:image/jpeg;base64,{{image2}}" width="300px">
                <img src="data:image/jpeg;base64,{{image3}}" width="300px">
            </body>
        </html>
        """

        return render_template_string(template, id=data_id, image1=image1_base64, image2=image2_base64, image3=image3_base64)

    return jsonify({'message': 'Data not found'}), 404

@app.route('/data/images/<data_id>/<image_number>', methods=['GET'])
def get_image_by_data_id(data_id, image_number):
    # Query Firestore based on the 'id' field in the data
    docs = db.collection('data').where('id', '==', data_id).stream()

    for doc in docs:  # This loop should only run once since 'id' should be unique
        data = doc.to_dict()

        if str(image_number) == '1':
            image_path = data['image1_path']
        elif str(image_number) == '2':
            image_path = data['image2_path']
        elif str(image_number) == '3':
            image_path = data['image3_path']
        else:
            return jsonify({'error': 'Invalid image number. Choose image1, image2, or image3.'}), 400

        return send_file(image_path, mimetype='image/jpeg')

    return jsonify({'message': 'Data not found'}), 404


@app.route('/data/<id>', methods=['GET'])
def get_data(id):
    doc = db.collection('data').document(id).get()
    if doc.exists:
        return jsonify(doc.to_dict()), 200
    else:
        return jsonify({'message': 'Data not found'}), 404

@app.route('/data', methods=['GET'])
def get_all_data():
    docs = db.collection('data').stream()
    response = []
    for doc in docs:
        response.append(doc.to_dict())
    return jsonify(response), 200

@app.route('/data/type/<data_type>/class/<int:data_class>', methods=['GET'])
def get_data_by_type_and_class(data_type, data_class):
    # Query Firestore based on type and class
    docs = db.collection('data').where('type', '==', data_type).where('class', '==', data_class).stream()
    
    response = []
    for doc in docs:
        response.append(doc.to_dict())
    
    if response:
        return jsonify(response), 200
    else:
        return jsonify({'message': 'No data found for given type and class'}), 404

@app.route('/data/<id>', methods=['PUT'])
def update_data(id):
    data_type = request.form['type']
    data_class = request.form['class']
    save_path = os.path.join(UPLOAD_FOLDER, data_type, data_class)

    data = {
        'type': data_type,
        'class': data_class,
        'topic1': request.form['topic1'],
        'description1': request.form['description1'],
        'topic2': request.form['topic2'],
        'description2': request.form['description2'],
    }

    if 'image1' in request.files:
        image1_filename = secure_filename(request.files['image1'].filename)
        request.files['image1'].save(os.path.join(save_path, image1_filename))
        data['image1_path'] = os.path.join(save_path, image1_filename)

    if 'image2' in request.files:
        image2_filename = secure_filename(request.files['image2'].filename)
        request.files['image2'].save(os.path.join(save_path, image2_filename))
        data['image2_path'] = os.path.join(save_path, image2_filename)

    if 'image3' in request.files:
        image3_filename = secure_filename(request.files['image3'].filename)
        request.files['image3'].save(os.path.join(save_path, image3_filename))
        data['image3_path'] = os.path.join(save_path, image3_filename)

    db.collection('data').document(id).set(data, merge=True)
    return jsonify({'message': 'Data updated successfully'}), 200

@app.route('/data/<id>', methods=['DELETE'])
def delete_data(id):
    doc = db.collection('data').document(id).get()
    if doc.exists:
        data = doc.to_dict()

        # Delete images from local storage
        os.remove(data['image1_path'])
        os.remove(data['image2_path'])
        os.remove(data['image3_path'])

        # Delete document from Firestore
        db.collection('data').document(id).delete()

    return jsonify({'message': 'Data deleted successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)