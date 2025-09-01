import streamlit as st
import cv2
import os
import numpy as np
import mediapipe as mp
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from PIL import Image
import tempfile
import threading
SIGN_NAMES = ["Hi", "Hello", "Yes", "No", "Thank You", "Please", "Sorry", "Help", "Stop", "Goodbye"]
DATA_DIR = "./data"
NUM_IMAGES_PER_CLASS = 100
MODEL_PATH = "model.p"
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
st.set_page_config(
    page_title="Sign Language App",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #333;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.status-box {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #b6d4db;
    color: #0c5460;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
}
.prediction-result {
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    background-color: #e7f3ff;
    color: #0066cc;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)
if 'collecting' not in st.session_state:
    st.session_state.collecting = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'labels' not in st.session_state:
    st.session_state.labels = SIGN_NAMES
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'word_buffer' not in st.session_state:
    st.session_state.word_buffer = ""

def create_data_directories():
    """Create data directories for each sign"""
    for name in SIGN_NAMES:
        os.makedirs(os.path.join(DATA_DIR, name.replace(" ", "_")), exist_ok=True)

def load_model():
    """Load the trained model"""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model_dict = pickle.load(f)
                st.session_state.model = model_dict['model']
                st.session_state.labels = model_dict.get('labels', SIGN_NAMES)
            return True
        return False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def extract_landmarks(image):
    """Extract hand landmarks from image"""
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hands.close()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            if len(landmarks) == 42:
                return landmarks
    return None

def train_model():
    """Train the model with collected data"""
    with st.spinner("Extracting hand landmarks from images..."):
        data = []
        labels = []
        
        progress_bar = st.progress(0)
        total_signs = len(SIGN_NAMES)
        
        for idx, word in enumerate(SIGN_NAMES):
            folder = os.path.join(DATA_DIR, word.replace(" ", "_"))
            if not os.path.exists(folder):
                continue
                
            images = os.listdir(folder)
            for img_name in images:
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                landmarks = extract_landmarks(img)
                if landmarks:
                    data.append(landmarks)
                    labels.append(idx)
            
            progress_bar.progress((idx + 1) / total_signs)
        
        if len(data) == 0:
            st.error("No valid training data found! Please collect images first.")
            return False
            
        st.success(f"Extracted landmarks from {len(data)} images")
    
    with st.spinner("Training Random Forest model..."):
        X = np.array(data)
        y = np.array(labels)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        st.session_state.model = model
        st.success(f"Model trained successfully!")
        st.info(f"Validation Accuracy: {accuracy*100:.2f}%")
        st.info(f"Trained on {len(data)} samples")
        
        return True

def save_model():
    """Save the trained model"""
    if st.session_state.model is None:
        st.error("No model to save!")
        return False
        
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                'model': st.session_state.model, 
                'labels': SIGN_NAMES
            }, f)
        st.success(f"Model saved as {MODEL_PATH}")
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def predict_sign(landmarks):
    """Predict sign from landmarks"""
    if st.session_state.model is None:
        return None
        
    try:
        prediction = st.session_state.model.predict([np.array(landmarks)])
        return st.session_state.labels[int(prediction[0])]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None
def main():
    st.markdown('<h1 class="main-header">🤟 Sign Language Recognition System</h1>', unsafe_allow_html=True)
    create_data_directories()
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Data Collection", "Model Training", "Sign Recognition"])
    
    if page == "Data Collection":
        data_collection_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Sign Recognition":
        sign_recognition_page()

def data_collection_page():
    st.markdown('<h2 class="section-header">📸 Data Collection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        camera_placeholder = st.empty()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            selected_word = st.selectbox("Select Sign:", SIGN_NAMES)
        with col_b:
            collect_btn = st.button("Start Collecting", disabled=st.session_state.collecting)
        with col_c:
            stop_btn = st.button("Stop Collecting", disabled=not st.session_state.collecting)
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
    
    with col2:
        st.subheader("Collection Status")
        for word in SIGN_NAMES:
            folder = os.path.join(DATA_DIR, word.replace(" ", "_"))
            count = len(os.listdir(folder)) if os.path.exists(folder) else 0
            st.metric(word, f"{count}/{NUM_IMAGES_PER_CLASS}")
    if collect_btn and not st.session_state.collecting:
        st.session_state.collecting = True
        st.rerun()
    
    if stop_btn and st.session_state.collecting:
        st.session_state.collecting = False
        st.rerun()
    if st.session_state.collecting:
        cap = cv2.VideoCapture(0)
        word = selected_word.replace(" ", "_")
        save_path = os.path.join(DATA_DIR, word)
        current_count = len(os.listdir(save_path))
        
        status_placeholder.info(f"Collecting images for '{selected_word}'... Click 'Stop Collecting' to stop.")
        
        if current_count < NUM_IMAGES_PER_CLASS:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                filename = os.path.join(save_path, f"{current_count}.jpg")
                cv2.imwrite(filename, frame)
                
                progress_placeholder.progress((current_count + 1) / NUM_IMAGES_PER_CLASS)
                time.sleep(0.1) 
        else:
            status_placeholder.success(f"Collection complete for '{selected_word}'!")
            st.session_state.collecting = False
        
        cap.release()
        
        if st.session_state.collecting:
            time.sleep(0.1)
            st.rerun()

def model_training_page():
    st.markdown('<h2 class="section-header">🧠 Model Training</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data Overview")
        total_images = 0
        data_status = {}
        
        for word in SIGN_NAMES:
            folder = os.path.join(DATA_DIR, word.replace(" ", "_"))
            count = len(os.listdir(folder)) if os.path.exists(folder) else 0
            total_images += count
            data_status[word] = count
            if count >= NUM_IMAGES_PER_CLASS:
                st.success(f"✅ {word}: {count} images")
            elif count > 0:
                st.warning(f"⚠️ {word}: {count} images (need {NUM_IMAGES_PER_CLASS - count} more)")
            else:
                st.error(f"❌ {word}: No images")
        
        st.info(f"Total images collected: {total_images}")
        
        # Training button
        if st.button("🚀 Start Training", disabled=total_images == 0):
            if train_model():
                st.balloons()
    
    with col2:
        st.subheader("Model Status")
        model_exists = os.path.exists(MODEL_PATH)
        if model_exists:
            st.success("✅ Trained model found")
            model_size = os.path.getsize(MODEL_PATH) / 1024  # KB
            st.info(f"Model size: {model_size:.1f} KB")
            
            if st.button("💾 Save Current Model"):
                save_model()
        else:
            st.warning("⚠️ No trained model found")
        if st.button("📂 Load Model"):
            if load_model():
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model")

def sign_recognition_page():
    st.markdown('<h2 class="section-header">👋 Sign Recognition</h2>', unsafe_allow_html=True)
    if st.session_state.model is None:
        if not load_model():
            st.error("❌ No trained model found! Please train a model first.")
            st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Recognition")
        camera_placeholder = st.empty()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            start_btn = st.button("🎥 Start Recognition", disabled=st.session_state.recognition_active)
        with col_b:
            stop_btn = st.button("⏹️ Stop Recognition", disabled=not st.session_state.recognition_active)
        with col_c:
            clear_btn = st.button("🗑️ Clear History")
        prediction_placeholder = st.empty()
    
    with col2:
        st.subheader("Recognition Results")
        st.text_area("Word Buffer:", st.session_state.word_buffer, height=100, disabled=True)
        st.subheader("Recent Predictions")
        if st.session_state.prediction_history:
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):
                st.text(f"{len(st.session_state.prediction_history)-i}: {pred}")
        else:
            st.text("No predictions yet")
    if start_btn:
        st.session_state.recognition_active = True
        st.rerun()
    
    if stop_btn:
        st.session_state.recognition_active = False
        st.rerun()
    
    if clear_btn:
        st.session_state.prediction_history = []
        st.session_state.word_buffer = ""
        st.rerun()
    if st.session_state.recognition_active:
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            current_prediction = "No hand detected"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])
                    
                    if len(landmarks) == 42:
                        predicted_sign = predict_sign(landmarks)
                        if predicted_sign:
                            current_prediction = predicted_sign
                            if (not st.session_state.prediction_history or 
                                st.session_state.prediction_history[-1] != predicted_sign):
                                st.session_state.prediction_history.append(predicted_sign)
                                st.session_state.word_buffer += predicted_sign + " "
            
            
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            prediction_placeholder.markdown(
                f'<div class="prediction-result">{current_prediction}</div>', 
                unsafe_allow_html=True
            )
        hands.close()
        cap.release()
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()
