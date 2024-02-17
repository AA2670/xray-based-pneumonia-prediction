from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import torch
from torchvision.transforms import transforms
from main import ConvNet  # Assuming your model is defined in a file named model.py

app = Flask(__name__, static_folder='static')
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

# Load the trained model
model_path = 'model_xray_new.ckpt'
model = ConvNet(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transformer = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']

    # Check if the file is allowed
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    if not (image_file.filename.lower().endswith(tuple(allowed_extensions)) and image_file.content_type.startswith('image/')):
        return jsonify({'error': 'Invalid file format'})

    # Save the image to a temporary file
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)

    # Load the image, apply transformations, and make the prediction
    image = Image.open(temp_path).convert('RGB')
    image = transformer(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
    
    _, predicted_class = torch.max(output.data, 1)
    
    # Clean up the temporary file
    os.remove(temp_path)

    # Return the prediction result
    prediction = "Pneumonia" if predicted_class.item() == 1 else "Normal"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
