from flask import Flask, render_template, request, jsonify, Response
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
cough_model = load_model('cough_detection_model.h5')
mask_model = load_model('mask_detection_model1.h5')  # مدل تشخیص ماسک
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_mfcc(audio_file):
    y, sr = librosa.load(audio_file, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=27)
    if mfcc.shape[1] < 100:
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :100]
    return mfcc

def create_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.imshow(S_DB, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def detect_mask(face_img):
    face = cv2.resize(face_img, (150, 150))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)
    pred = mask_model.predict(face)[0][0]
    return pred > 0.5

@app.route('/check_mask', methods=['POST'])
def check_mask():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty image file'}), 400

        image_data = image_file.read()
        if not image_data:
            return jsonify({'error': 'Image data is empty'}), 400

        nparr = np.frombuffer(image_data, np.uint8)
        if nparr.size == 0:
            return jsonify({'error': 'Invalid or empty image data'}), 400

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        mask_found = False

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if detect_mask(face_img):
                mask_found = True
                break

        return jsonify({'mask_detected': mask_found})

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'trace': traceback.format_exc()
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    try:
        audio_file = request.files['audio']
        audio_file.seek(0)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'recording.wav')
        audio_file.save(filename)
        
        if not os.path.exists(filename):
            return jsonify({'error': 'File save failed'}), 500

        mfcc = extract_mfcc(filename)
        mfcc_input = np.expand_dims(mfcc, axis=0)
        pred = cough_model.predict(mfcc_input)
        final_pred = pred[0][0]

        result = final_pred > 0.7
        spectrogram = create_spectrogram(filename)

        return jsonify({
            'result': 'Infected with COVID-19' if result else 'Healthy',
            'probability': float(final_pred),
            'spectrogram': spectrogram
        })

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'trace': traceback.format_exc()
        }), 500
    
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)