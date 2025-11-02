# live_training_dashboard.py

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import threading
from flask import Flask, render_template_string, jsonify
import plotly.graph_objects as go
import tqdm

# ----------------------------
# Paths and parameters
# ----------------------------
base_folder = '/Users/aakriti/AcousticProcessing/SoundWaves'
output_folder = '/Users/aakriti/AcousticProcessing/Spectrogram'

sr = 22050
segment_duration = 4
overlap = 0.5
n_mels = 128

# ----------------------------
# Helper functions
# ----------------------------
def segments(y, sr, segment_duration=4, overlap=0.5):
    segment_length = int(sr * segment_duration)
    hop = int(segment_length * (1 - overlap))
    segs = []
    for start in range(0, len(y) - segment_length + 1, hop):
        end = start + segment_length
        segs.append(y[start:end])
    if not segs:
        y_padded = np.pad(y, (0, segment_length - len(y)))
        segs = [y_padded]
    return segs

def convert_segments(segment, sr=sr, n_mels=n_mels):
    mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# ----------------------------
# Prepare dataset
# ----------------------------
os.makedirs(output_folder, exist_ok=True)

X, y_labels = [], []
labels = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

for label_index, class_name in enumerate(labels):
    class_folder = os.path.join(base_folder, class_name)
    for file in tqdm.tqdm(os.listdir(class_folder)):
        if not file.lower().endswith(('.wav', '.mp3', '.flac')):
            continue
        file_path = os.path.join(class_folder, file)
        try:
            audio, _ = librosa.load(file_path, sr=sr)
            segs = segments(audio, sr, segment_duration, overlap)
            for seg in segs:
                mel_db = convert_segments(seg, sr, n_mels)
                mel_db = (mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db))
                mel_db = np.expand_dims(mel_db, axis=-1)
                X.append(mel_db)
                y_labels.append(label_index)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

X = np.array(X)
y_labels = np.array(y_labels)
print(f"Loaded {len(X)} spectrograms")

# ----------------------------
# Split data
# ----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y_labels, test_size=0.3, stratify=y_labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ----------------------------
# Define model
# ----------------------------
input_shape = X_train.shape[1:]
num_classes = len(labels)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ----------------------------
# Live dashboard setup
# ----------------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<h1 style="text-align:center;">Training Metrics</h1>
<div id="graph"></div>
<script>
function fetchAndUpdate() {
    fetch('/metrics')
    .then(res => res.json())
    .then(json => {
        Plotly.react('graph', json.data, json.layout);
    });
}
fetchAndUpdate();
setInterval(fetchAndUpdate, 2000);
</script>
</body>
</html>
"""

def make_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines+markers', name='Train Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=val_losses, mode='lines+markers', name='Val Loss', line=dict(color='orange')))
    fig.add_trace(go.Scatter(y=train_accs, mode='lines+markers', name='Train Acc', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=val_accs, mode='lines+markers', name='Val Acc', line=dict(color='red')))
    fig.update_layout(title="Training & Validation Metrics",
                      xaxis_title="Epoch", yaxis_title="Value",
                      template="plotly_white", width=1000, height=600)
    return fig.to_dict()

@app.route("/")
def index():
    fig = make_figure()
    return render_template_string(HTML_TEMPLATE, data=fig['data'], layout=fig['layout'])

@app.route("/metrics")
def metrics():
    fig = make_figure()
    return jsonify(fig)

def run_app():
    app.run(debug=False, use_reloader=False)

# Start dashboard in a separate thread
threading.Thread(target=run_app).start()

# ----------------------------
# Training loop
# ----------------------------
epochs = 20
batch_size = 32

for epoch in range(epochs):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=batch_size,
        verbose=1
    )

    train_losses.append(history.history['loss'][0])
    val_losses.append(history.history['val_loss'][0])
    train_accs.append(history.history['accuracy'][0])
    val_accs.append(history.history['val_accuracy'][0])

    print(f"Epoch {epoch+1}/{epochs} completed.")

print("Training finished. Open http://127.0.0.1:5000/ in your browser to see the live dashboard.")
