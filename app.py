from flask import Flask, request, jsonify, send_from_directory
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from CNN import FasterCNN

app = Flask(__name__)

# Load the trained model
num_classes = 15  
model = FasterCNN(num_classes)  
model.load_state_dict(torch.load("leaf-disease-detection.pth", map_location=torch.device("cpu")))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) 
])

# Class labels
class_labels = [
    "Pepper bell Bacterial spot", "Pepper bell healthy", "Potato Early blight",
     "Potato Late blight","Potato healthy","Tomato Bacterial spot","Tomato Early blight",
       "Tomato Late blight", "Tomato Leaf Mold",
       "Tomato Septoria leaf spot","Tomato Spider mites Two spotted spider mite",
       "Tomato Target Spot", "Tomato Tomato YellowLeaf Curl Virus",
    "Tomato Tomato mosaic virus","Tomato healthy"    
]

# Serve the HTML page
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")

# Handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class_index = output.argmax(dim=1).item()

    predicted_class_name = class_labels[predicted_class_index]

    return jsonify({"prediction": predicted_class_name})

if __name__ == "__main__":
    app.run(debug=True)