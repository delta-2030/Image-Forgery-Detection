#Import necessary libraries and packages
import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from skimage.feature import local_binary_pattern

# Extract frames from the video
def extract_frames_from_video(video_path, output_folder=r"E:\DIP Project\Deepfake\frames"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = video.read()
    while success:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        success, frame = video.read()
        frame_count += 1
    video.release()
    print(f"Extracted {frame_count} frames.")

# Detect Faces in frames using Haar cascade
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

#Process the frame only if face is detected in the frame
def process_frame_if_face_detected(frame_path):
    frame = cv2.imread(frame_path)
    if len(detect_faces(frame)) > 0: 
        return extract_features_from_image(frame_path)
    return None 

# Extract features from each from frame
def extract_features_from_image(image_path, prev_frame_path=None, fixed_length=1000):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(image, n_points, radius, method="uniform") # Texture information using Local Binary Pattern
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    edges = cv2.Canny(image, threshold1=100, threshold2=200)# Edge information using Canny's edge detection algorithm
    edges_flat = edges.flatten()
    if len(edges_flat) < fixed_length:
        edges_flat = np.pad(edges_flat, (0, fixed_length - len(edges_flat)), 'constant')
    else:
        edges_flat = edges_flat[:fixed_length]

    motion_features = []
    if prev_frame_path is not None: #If previous frame is provided compute motion features
        prev_frame = cv2.imread(prev_frame_path)
        motion_score = analyze_motion(prev_frame, cv2.imread(image_path))
        motion_features = [motion_score]

    artifact_features = detect_deepfake_artifacts(cv2.imread(image_path)) # Extracts some  other deepfake artifacts

    features = np.concatenate((lbp_hist, edges_flat, artifact_features, motion_features))# Combines all features into a single vector
    if len(features) < fixed_length:
        features = np.pad(features, (0, fixed_length - len(features)), 'constant')
    else:
        features = features[:fixed_length]

    return features


def analyze_motion(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    prvs = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)# Motion analysis using Farneback algorithm
    magnitude, angle = cv2.cartToPolar(prvs[..., 0], prvs[..., 1])
    motion_score = np.mean(magnitude)

    return motion_score


def detect_deepfake_artifacts(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurriness_score = cv2.Laplacian(gray, cv2.CV_64F).var() # Bluriness
    mean_color = np.mean(image, axis=(0, 1))# Color statistics
    std_color = np.std(image, axis=(0, 1))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv_image[:, :, 1]) # Saturation
    mean_hue = np.mean(hsv_image[:, :, 0]) # Hue
    artifact_features = np.array([
        blurriness_score,
        mean_color[0], mean_color[1], mean_color[2],
        std_color[0], std_color[1], std_color[2],
        mean_saturation,
        mean_hue
    ])
    return artifact_features


def prepare_data(video_paths, labels, output_folder=r"E:\DIP Project\Deepfake\frames", fixed_length=1000):
    X = []
    y = []
    for video_path, label in zip(video_paths, labels):
        extract_frames_from_video(video_path, output_folder)
        frame_files = sorted(os.listdir(output_folder))
        prev_frame_path = None
        for frame_file in frame_files:
            frame_path = os.path.join(output_folder, frame_file)
            features = process_frame_if_face_detected(frame_path)  
            if features is not None:  
                X.append(features)
                y.append(label)
                prev_frame_path = frame_path
    X = np.array([np.array(f) for f in X])
    y = np.array(y)
    return X, y


def train_deepfake_detector(real_videos, fake_videos, model_path=r"E:\DIP Project\Deepfake\deepfake_classifier.pkl"):
    labels = [0] * len(real_videos) + [1] * len(fake_videos)
    video_paths = real_videos + fake_videos
    fixed_length = 1000  # Define a fixed feature vector length for consistency
    X, y = prepare_data(video_paths, labels, fixed_length=fixed_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into test and training sets

    classifier = RandomForestClassifier(n_estimators=100, random_state=42) # Load a RandomForestClassifier model
    classifier.fit(X_train, y_train) # Train the model on Real and Fake videos

    y_pred = classifier.predict(X_test)
    joblib.dump(classifier, model_path) # Save the model in model_path
    print(f"Model saved to {model_path}")

real_videos = [r"E:\DIP Project\Deepfake\Real videos\test1.mp4"]
fake_videos = [r"E:\DIP Project\Deepfake\Deepfaked\download.mp4"]

train_deepfake_detector(real_videos, fake_videos)
