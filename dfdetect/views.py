from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.core.mail import send_mail
from .models import OTP

from django.shortcuts import render
from django.http import JsonResponse
import subprocess

import os
import subprocess
from django.http import JsonResponse
from django.conf import settings

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.core.mail import send_mail
from .models import OTP

from django.http import JsonResponse

def login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            return redirect('index')  # Redirect to OTP verification page
        else: 
            return redirect('login')

    return render(request, 'login.html')

def index(request):
    return render(request, 'index.html')

def detect(request):
    return render(request, 'detect.html')

def signup(request):
    if request.method == "POST":
        email = request.POST.get('email','')
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # Check if passwords match
        if password != confirm_password:
            return render(request, 'signup.html', {'error': 'Passwords do not match'})

        # Validate password (8+ characters, including a number and symbol)
        if not (any(char.isdigit() for char in password) and any(char in "!@#$%^&*()" for char in password) and len(password) >= 8):
            return render(request, 'signup.html', {'error': 'Password must be 8+ characters, including a number and symbol'})

        # Check if user already exists
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html', {'error': 'User already exists. Please login.'})

        # Create user
        user = User.objects.create_user(username=username, email=email, password=password)
        user.is_active = False  # User cannot log in until OTP is verified
        user.save()

        # Generate OTP and send email
        otp_instance = OTP.objects.create(user=user)
        otp_code = otp_instance.generate_otp()
        otp_instance.otp_code = otp_code
        otp_instance.save()

        send_mail(
            'Your OTP Code',
            f'Your OTP is {otp_code}. Please enter it to verify your account.',
            'sindhukvs45@gmail.com',
            [email],
            fail_silently=False,
        )

        return render(request, 'verify_otp.html', {'email': email})

    return render(request, 'signup.html')

def verify_otp(request):
    if request.method == "POST":
        email = request.POST.get('email')
        entered_otp = request.POST.get('otp')

        try:
            user = User.objects.get(email=email)
            otp_instance = OTP.objects.get(user=user)

            if otp_instance.otp_code == entered_otp:
                user.is_active = True
                user.save()
                otp_instance.delete()
                return redirect('login')  # Assuming 'login' is the name of your login URL pattern

            else:
                return render(request, 'verify_otp.html', {'error': 'Invalid OTP', 'email': email})

        except (User.DoesNotExist, OTP.DoesNotExist):
            return render(request, 'verify_otp.html', {'error': 'Invalid request'})

    return render(request, 'verify_otp.html')





import cv2
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse

from .models import UploadedVideo
from .forms import VideoUploadForm
from facenet_pytorch import MTCNN

# -------------------------------
# Define your Deepfake model
# -------------------------------
class DeepfakeModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DeepfakeModel, self).__init__()
        # Use a pretrained ResNeXt-50 with the updated weights parameter
        backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        # Note: In your checkpoint, the CNN backbone was saved under the name "model"
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        # The training checkpoint used the name "linear1" for the final layer.
        self.linear1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, C, H, W = x.shape
        x = x.view(batch_size * seq_length, C, H, W)
        features = self.model(x)
        x = self.avgpool(features)
        x = x.view(batch_size, seq_length, -1)  # now x has shape (batch_size, seq_length, latent_dim)
        lstm_out, _ = self.lstm(x)
        # Average the LSTM outputs over the time dimension
        lstm_out = torch.mean(lstm_out, dim=1)
        out = self.dropout(lstm_out)
        out = self.linear1(out)
        return out

# -------------------------------------------
# Load the saved model checkpoint
# -------------------------------------------



# -------------------------------------------
# Helper Functions
# -------------------------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 99:  # limit frames for performance
            break
        frame_count += 1
        frames.append(frame)
    cap.release()
    return frames

def predict_deepfake(faces):
    predictions = []
    print(len(faces))
    for face in faces:
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue  # Skip invalid detections
        # Create a tensor for a single face image
        face_tensor = transform(face).unsqueeze(0)  # shape: (1, C, H, W)
        # Add a sequence dimension so that shape becomes (1, 1, C, H, W)
        face_tensor = face_tensor.unsqueeze(1)
        with torch.no_grad():
            output = model(face_tensor)
        # Convert output logits to probability using softmax
        prob = torch.softmax(output, dim=1)[0][1].item()  # probability for FAKE class
        predictions.append(prob)
    
    if predictions:
        avg_prediction = np.mean(predictions)
        if avg_prediction > 0.6:
            label="REAL"
        else:
            label="FAKE"
        confidence = round((avg_prediction * 100), 2)
    else:
        label, confidence = "UNKNOWN", 0.0
    print(predictions)
    
    
    return label, confidence


# -------------------------------------------
# Django View
# -------------------------------------------
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_instance = form.save()
            video_path = default_storage.save(
                video_instance.video_file.name,
                ContentFile(video_instance.video_file.read())
            )
            video_full_path = os.path.join(default_storage.location, video_path)
            heatmap_base64 = generate_heatmap(video_full_path)
            frames = extract_frames(video_full_path)
            faces = []
            encoded_frames = []
            
            for frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        face = frame_rgb[y1:y2, x1:x2]
                        faces.append(face)
                pil_im = Image.fromarray(frame_rgb)
                buff = BytesIO()
                pil_im.save(buff, format="JPEG")
                img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
                encoded_frames.append(img_str)
            
            label, confidence = predict_deepfake(faces)
            
            return JsonResponse({
                'status': 'success',
                'label': label,
                'confidence': confidence,
                'frames': encoded_frames,
                'heatmap':heatmap_base64,
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors})
    else:
        form = VideoUploadForm()
    return render(request, 'upload_video.html', {'form': form})

import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_heatmap(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    heatmap = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize or accumulate the heatmap
        if heatmap is None:
            heatmap = np.zeros_like(gray, dtype=np.float32)
        heatmap += gray.astype(np.float32)

    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)

    # Apply a colormap to the heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert the heatmap to a base64-encoded image
    _, buffer = cv2.imencode('.png', heatmap_colored)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    return heatmap_base64



import cv2
import ffmpeg
import os
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

import cv2
import os
from pymediainfo import MediaInfo
import os
import datetime
def get_video_metadata(video_path):
    """Extracts detailed metadata from a video file without FFmpeg."""
    if not os.path.exists(video_path):
        return "File not found."

    # Extract metadata using PyMediaInfo
    media_info = MediaInfo.parse(video_path)
    video_track = next((track for track in media_info.tracks if track.track_type == "Video"), None)

    # Extract metadata using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video file."

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate if frame_rate else "Unknown"

    cap.release()
    try:
        file_size = os.path.getsize(video_path)
    except Exception as e:
        file_size = None

    # Get file creation date (platform-dependent; on Unix, it might be the last metadata change)
    try:
        creation_timestamp = os.path.getctime(video_path)
        creation_date = datetime.datetime.fromtimestamp(creation_timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        creation_date = None
    # Construct metadata dictionary
    metadata = {
        "Width": int(width),  # Convert NumPy int to Python int
    "Height": int(height),
    "Duration (seconds)": float(duration),  # Convert NumPy float to Python float
    "Frame Rate (FPS)": float(frame_rate),
    "Codec": str(video_track.format) if video_track else "Unknown",
    "Bitrate": int(video_track.bit_rate) if video_track and video_track.bit_rate else "Unknown",
    "Pixel Format": str(video_track.pixel_format) if video_track else "Unknown",
    "Compression": str(video_track.compression_mode) if video_track else "Unknown",
    
    # Additional metadata items from the original expansion
    "Aspect Ratio": round(int(width) / int(height), 2) if int(height) != 0 else "Unknown",
    "Frame Count": int(float(duration) * float(frame_rate)),
    "Rotation": str(video_track.rotation) if video_track and hasattr(video_track, 'rotation') else "Unknown",
    "Color Space": str(video_track.color_space) if video_track and hasattr(video_track, 'color_space') else "Unknown",
    
    # Existing non-audio keys
    "File Size": int(file_size) if 'file_size' in locals() and file_size else "Unknown",  # in bytes
    "Creation Date": str(creation_date) if 'creation_date' in locals() and creation_date else "Unknown",
    
    # Three new metadata items
    "Container Format": str(video_track.container_format) if video_track and hasattr(video_track, 'container_format') else "Unknown",
    "Scan Type": str(video_track.scan_type) if video_track and hasattr(video_track, 'scan_type') else "Unknown",
    "Color Range": str(video_track.color_range) if video_track and hasattr(video_track, 'color_range') else "Unknown",

    }

    return  metadata

def get_image_metadata(image_path):
    """Extracts detailed metadata from an image file for deepfake detection."""
    if not os.path.exists(image_path):
        return "File not found."
    
    image = Image.open(image_path)
    exif_data = image._getexif()
    metadata = {
        "Format": image.format,
        "Mode": image.mode,
        "Size": image.size,
        "Compression": image.info.get("compression", "Unknown"),
        "DPI": image.info.get("dpi", "Unknown"),
        "ICC Profile": "Available" if "icc_profile" in image.info else "Not Available"
    }
    
    if exif_data:
        exif_metadata = {TAGS.get(tag, tag): value for tag, value in exif_data.items() if tag in TAGS}
        metadata.update(exif_metadata)
    
    return metadata

def check_metadata_anomalies(metadata):
    """Detects inconsistencies in metadata that could indicate deepfake manipulation."""
    anomalies = []
    if "DPI" in metadata and metadata["DPI"] != "Unknown":
        if isinstance(metadata["DPI"], tuple) and metadata["DPI"][0] < 72:
            anomalies.append("Low DPI may indicate digital tampering.")
    if "ICC Profile" in metadata and metadata["ICC Profile"] == "Not Available":
        anomalies.append("Missing ICC Profile may suggest image recompression.")
    if "Compression" in metadata and metadata["Compression"] == "Unknown":
        anomalies.append("No compression data may indicate altered metadata.")
    
    return anomalies if anomalies else "No anomalies detected."

def extract_video_frames(video_path, frame_interval=30):
    """Extracts frames from a video at a specified interval."""
    if not os.path.exists(video_path):
        return "File not found."
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    
    return frames

from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os

from django.http import JsonResponse
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import VideoUploadForm
#from .utils import get_video_metadata, extract_video_frames, get_image_metadata, check_metadata_anomalies

from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import VideoUploadForm
 # Ensure you import the right functions


import os
import json
import csv
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
from pymediainfo import MediaInfo
from .forms import ImageUploadForm  # Ensure you have an ImageUploadForm

# Function to extract metadata from a single image
from PIL import Image
from PIL.ExifTags import TAGS

def extract_image_metadata(image_path):
    metadata = {}

    try:
        with Image.open(image_path) as img:
            metadata["Filename"] = image_path.split("/")[-1]
            metadata["Size"] = f"{img.width}x{img.height}"
            metadata["Format"] = img.format
            metadata["Color Mode"] = img.mode
            metadata["MIME Type"] = Image.MIME.get(img.format, "Unknown")

            # Extract EXIF data
            exif_data = img.getexif()
            exif_info = {}

            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)  # Convert tag ID to name
                    exif_info[tag_name] = value

            # Store extracted EXIF or show "No Metadata Found"
            metadata["Info"] = exif_info if exif_info else "No EXIF metadata found"

    except Exception as e:
        metadata["Error"] = str(e)

    return metadata


# Function to handle image metadata extraction
def image_metadata(request):
    if request.method == 'POST':
        files = request.FILES.getlist("images")  # Accept multiple files
        form = ImageUploadForm(request.POST, request.FILES)

        if not files:
            return JsonResponse({"status": "error", "message": "No files selected."})

        metadata_list = []

        for file in files:
            try:
                # Save file temporarily
                file_path = default_storage.save(file.name, ContentFile(file.read()))
                full_path = os.path.join(default_storage.location, file_path)

                # Extract metadata
                metadata = extract_image_metadata(full_path)
                metadata_list.append(metadata)
            
            except Exception as e:
                return JsonResponse({"status": "error", "message": f"Error processing {file.name}: {str(e)}"})

        return JsonResponse({"status": "success", "metadata": metadata_list})

    return render(request, "imagemetadata.html", {"form": ImageUploadForm()})
# Function to download metadata as CSV
def download_image_metadata(request):
    metadata_json = request.GET.get("data")
    if not metadata_json:
        return JsonResponse({"status": "error", "message": "No metadata provided."})

    metadata_list = json.loads(metadata_json)

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="metadata.csv"'

    writer = csv.writer(response)
    writer.writerow(["Filename", "Format", "Mode", "Size", "Color Mode", "MIME Type", "Info"])

    for metadata in metadata_list:
        writer.writerow([
            metadata.get("Filename", ""),
            metadata.get("Format", ""),
            metadata.get("Mode", ""),
            metadata.get("Size", ""),
            metadata.get("Color Mode", ""),
            metadata.get("MIME Type", ""),
            metadata.get("Info", ""),
        ])

    return response


import numpy as np
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load a pre-trained model (VGG16)
model2 = VGG16(weights="imagenet")

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Overlay the Grad-CAM heatmap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageChops, ImageEnhance
# Preprocess the image for the model
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize if required by the model
    return img
def preprocess_img(img_path, target_size=(224, 224)):  # Ensure correct size
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)  # Normalize for model

# Generate Grad-CAM heatmap
def grad_cam(model, image, class_idx, target_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(image)  # Watch image for gradient tracking
        conv_outputs, predictions = grad_model(image)

        # Handle binary classification
        loss = predictions[:, 0]+ 1e-8 if predictions.shape[1] == 1 else predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    grads=tf.abs(grads)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # Corrected axis
    heatmap = tf.reduce_sum(weights * conv_outputs[0], axis=-1)
    heatmap = np.maximum(heatmap,1e-8)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize
    heatmap = np.uint8(255 * heatmap)  
    return heatmap



def overlay_heatmap(image_path, alpha=0.4, target_size=(256, 256)):
    # Load image and resize to model's expected input size
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    # Get model prediction
        img_input = np.expand_dims(img / 255.0, axis=0)  # Normalize and add batch dim
        prediction = model.predict(img_input)
        
        # Ensure prediction is extracted properly for binary classification
        pred_class = 1 if prediction[0] == 1  else 0
        pred_prob = float(prediction[0]) if pred_class == 1 else 1 - float(prediction[0])
        print(prediction)
    label = 'Authentic' if pred_class == 1 else 'Forged'
    print(f"Prediction: {label} with {pred_prob * 100:.2f}% confidence")
    return label,formatted_prob
def grad_cam2(model, img_array, layer_name="block5_conv3"):
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])  # Target class index
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)  # Compute gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling
    conv_outputs = conv_outputs.numpy()[0]

    # Compute weighted activation map
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # Apply ReLU
    heatmap /= np.max(heatmap)  # Normalize
    
    return heatmap
def overlay_heatmap2(img_path, heatmap, alpha=0.5):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)  # Convert to RGB scale
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map
    
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)  # Overlay heatmap
    return encode_image_to_base64(superimposed_img)
def error_level_analysis(image_path, quality=90, enhance_factor=10):
    # Load the original image
    original = Image.open(image_path).convert("RGB")
    
    # Save a recompressed version at lower quality
    recompressed_path = "recompressed.jpg"
    original.save(recompressed_path, "JPEG", quality=quality)
    recompressed = Image.open(recompressed_path)

    # Compute the ELA difference
    ela_image = ImageChops.difference(original, recompressed)

    # Enhance the contrast for better visibility
    enhancer = ImageEnhance.Contrast(ela_image)
    ela_image = enhancer.enhance(enhance_factor)

    # Display the ELA heatmap
    plt.figure(figsize=(8, 8))
    
    plt.axis("off")
    plt.title("Error Level Analysis (ELA)")
    



def encode_image_to_base64(img):
    """Convert OpenCV image to Base64 format"""
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


def overlayheatmap(image_path, heatmap,alpha=0.4, target_size=(256, 256)):
    # Load image and resize to model's expected input size
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    
    model=load_model(r"C:\Users\sindh\Downloads\140k_best_model.keras")
    # Get model prediction
    img_input = np.expand_dims(img / 255.0, axis=0)  # Normalize and add batch dim
    prediction = model.predict(img_input)
    
    # Ensure prediction is extracted properly for binary classification
    pred_class = 1 if prediction[0] == 1  else 0
    pred_prob = float(prediction[0]) if pred_class == 1 else 1 - float(prediction[0])
    print(prediction)
    label = 'Real' if pred_class == 1 else 'Fake'
    print(f"Prediction: {label} with {pred_prob * 100:.2f}% confidence")
    return label,pred_prob*100

def imagedetect(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_instance = form.save()
            video_path = default_storage.save(
                video_instance.video_file.name,
                ContentFile(video_instance.video_file.read())
            )
            
            image_path = os.path.join(default_storage.location, video_path)
            target_layer_name = "conv2_block6_concat"  # Modify based on model architecture
           
            preprocessed_image = preprocess_image(image_path, target_size=(256, 256))
            preprocessed_image = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)

            if len(preprocessed_image.shape) == 3:  # (height, width, channels)
                preprocessed_image = tf.expand_dims(preprocessed_image, axis=0) 
            class_idx = 1

            model = load_model("/content/sample_data/140k_best_model.keras", compile=False)
            heatmap = grad_cam(model, preprocessed_image, class_idx, target_layer_name)
            label,confidence=overlay_heatmap( image_path,heatmap)

            reprocessed_image = preprocess_img(image_path)
            heatmap2 = grad_cam2(model2, reprocessed_image)

            # Overlay heatmap on the original image
            heatmap_base64= overlay_heatmap2(image_path, heatmap2)
            
            #error_level_analysis(image_path)
            
            
            return JsonResponse({
                'status': 'success',
                'label': label,
                'confidence': confidence,
                "image_data": heatmap_base64,
            })
        else:
            return JsonResponse({'status': 'error', 'errors': form.errors})
    else:
        form = VideoUploadForm()
    return render(request, 'imagedetect.html', {'form': form})


import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import Video
from moviepy.editor import VideoFileClip
import csv

def upload_videos(request):
    if request.method == 'POST' and request.FILES.getlist('videos1'):
 
        for file in request.FILES.getlist('videos1'):
            Video.objects.create(file=file)
        return redirect('metadatavideo')
    return render(request, 'videometa.html')


from django.shortcuts import render
from .models import Video
from moviepy.editor import VideoFileClip

from django.http import JsonResponse

def metadatavideo(request):
   if request.method == 'POST' and request.FILES.getlist('videos1'):
        metadata_list = []
        for file in request.FILES.getlist('videos1'):
            video = Video(file=file)
            video.save()
            try:
                clip = VideoFileClip(video.file.path)
                metadata = {
                    'filename': file.name,
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'size': clip.size,
                    'audio': clip.audio is not None,
                }
                metadata_list.append(metadata)
                clip.close()
            except Exception as e:
                metadata_list.append({'error': str(e)})
        return JsonResponse({'status': 'success', 'metadatavideo': metadata_list[0]})  # Return JSON
   return render(request, 'videometa.html')
 
def download_metadata(request):
    videos = Video.objects.all()
    metadata_list = []
    for video in videos:
        try:
            clip = VideoFileClip(video.file.path)
            metadata = {
                'filename': os.path.basename(video.file.name),
                'duration': clip.duration,
                'fps': clip.fps,
                'size': clip.size,
                'audio': clip.audio is not None,
            }
            metadata_list.append(metadata)
            clip.close()
        except Exception as e:
            metadata_list.append({'error': str(e)})
    # Create a CSV file
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="video_metadata.csv"'
    writer = csv.writer(response)
    writer.writerow(['Filename', 'Duration (s)', 'FPS', 'Resolution', 'Has Audio'])
    for metadata in metadata_list:
        writer.writerow([
            metadata.get('filename', 'N/A'),
            metadata.get('duration', 'N/A'),
            metadata.get('fps', 'N/A'),
            f"{metadata.get('size', ('N/A', 'N/A'))[0]}x{metadata.get('size', ('N/A', 'N/A'))[1]}",
            metadata.get('audio', 'N/A'),
        ])
    return response


