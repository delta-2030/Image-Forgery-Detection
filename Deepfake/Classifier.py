import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
def extract_frames_from_video(video_path, output_folder="frames"):
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
def extract_features_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1 
    n_points = 8 * radius 
    lbp_image = local_binary_pattern(image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6) 
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    edges_flat = edges.flatten()
    target_length = lbp_hist.shape[0]
    if len(edges_flat) < target_length:
        edges_flat = np.pad(edges_flat, (0, target_length - len(edges_flat)), 'constant')
    elif len(edges_flat) > target_length:
        edges_flat = edges_flat[:target_length]
    features = np.concatenate((lbp_hist, edges_flat))
    return features
def prepare_data(video_path, label, output_folder="frames"):
    X = [] 
    y = []  
    extract_frames_from_video(video_path, output_folder)
    frame_files = os.listdir(output_folder)
    for frame_file in frame_files:
        frame_path = os.path.join(output_folder, frame_file)
        features = extract_features_from_image(frame_path)
        X.append(features)
        y.append(label)  
    
    return np.array(X), np.array(y)
def train_deepfake_classifier(real_video_path, fake_video_path):
    X_real, y_real = prepare_data(real_video_path, 0)  
    X_fake, y_fake = prepare_data(fake_video_path, 1)  
    X = np.vstack((X_real, X_fake))
    y = np.hstack((y_real, y_fake))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    joblib.dump(clf, 'deepfake_classifier.pkl')
    print("Model saved as 'deepfake_classifier.pkl'")


real_video_path = "real_video.mp4"  
fake_video_path = "download.mp4"  

train_deepfake_classifier(real_video_path, fake_video_path)
