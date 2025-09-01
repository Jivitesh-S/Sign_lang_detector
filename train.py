import customtkinter as ctk
from tkinter import messagebox
import cv2
import os
import threading
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
SIGN_NAMES = ["Hi", "Hello", "Yes", "No", "Thank You", "Please", "Sorry", "Help", "Stop", "Goodbye"]
DATA_DIR = "./data"
NUM_IMAGES_PER_CLASS = 100
MODEL_PATH = "model.p"
class SignTrainerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language - Collect & Train")
        self.geometry("1140x680")
        self.resizable(False, False)
        self.cap = cv2.VideoCapture(0)
        self.collecting = False
        self.counter = 0
        self.status_text = ctk.StringVar(value="Ready")
        self.training_status = ctk.StringVar(value="")
        self.accuracy_text = ctk.StringVar(value="")
        left_panel = ctk.CTkFrame(self, width=420, height=610, corner_radius=17)
        left_panel.pack(side="left", padx=30, pady=23)
        left_panel.pack_propagate(False)
        ctk.CTkLabel(left_panel, text="Live Camera Preview", font=("Segoe UI SemiBold", 20)).pack(pady=(17,7))
        self.video_label = ctk.CTkLabel(left_panel, text="", width=388, height=320, corner_radius=12)
        self.video_label.pack()
        colframe = ctk.CTkFrame(left_panel, fg_color="#eef4fc", corner_radius=10)
        colframe.pack(fill="x", pady=(21,7), padx=2)
        ctk.CTkLabel(colframe, text="Choose Word:", font=("Arial", 15, "bold")).pack(side="left", padx=(8,7))
        self.word_var = ctk.StringVar(value=SIGN_NAMES[0])
        self.word_menu = ctk.CTkOptionMenu(colframe, variable=self.word_var, values=SIGN_NAMES, width=128)
        self.word_menu.pack(side="left", padx=7)
        self.collect_btn = ctk.CTkButton(colframe, text="Start Collecting", command=self.start_collection, width=135)
        self.collect_btn.pack(side="left", padx=8)
        self.stop_btn = ctk.CTkButton(colframe, text="Stop", command=self.stop_collection, width=63, state="disabled")
        self.stop_btn.pack(side="left", padx=7)
        self.collect_status_lbl = ctk.CTkLabel(left_panel, textvariable=self.status_text, font=("Arial", 14, "italic"))
        self.collect_status_lbl.pack(pady=(6, 0))
        self.count_lbl = ctk.CTkLabel(left_panel, text="0/100 images collected", font=("Arial", 14, "bold"))
        self.count_lbl.pack(pady=(11, 0))
        right_panel = ctk.CTkFrame(self, width=612, height=610, corner_radius=17)
        right_panel.pack(side="right", padx=18, pady=23)
        right_panel.pack_propagate(False)
        ctk.CTkLabel(right_panel, text="Train Model", font=("Segoe UI Semibold", 22)).pack(pady=(23, 9))
        self.train_btn = ctk.CTkButton(right_panel, text="Start Training", command=self.train_model, width=170)
        self.train_btn.pack(pady=(6, 7))
        self.training_lbl = ctk.CTkLabel(right_panel, textvariable=self.training_status, font=("Arial", 15, "italic"))
        self.training_lbl.pack()
        ctk.CTkLabel(right_panel, text="Accuracy (validation):", font=("Arial", 16)).pack(pady=(13,5))
        self.acc_lbl = ctk.CTkLabel(right_panel, textvariable=self.accuracy_text, font=("Arial", 17, "bold"), text_color="#185c13")
        self.acc_lbl.pack(pady=(1, 13))
        self.save_btn = ctk.CTkButton(right_panel, text="Save Model", command=self.save_model, width=130, state="disabled")
        self.save_btn.pack()
        ctk.CTkLabel(right_panel, text="Status:", font=("Arial", 15, "bold")).pack(pady=(34,3))
        self.training_status_lbl = ctk.CTkLabel(right_panel, text="", font=("Arial", 13))
        self.training_status_lbl.pack(pady=(0,11))
        self.data = []
        self.labels = []
        self.model = None
        self.label_map = SIGN_NAMES
        for name in SIGN_NAMES:
            os.makedirs(os.path.join(DATA_DIR, name.replace(" ", "_")), exist_ok=True)
        self.update_video()

    def start_collection(self):
        self.collecting = True
        self.counter = 0
        self.collect_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_text.set(f"Collecting for '{self.word_var.get()}'...")
        threading.Thread(target=self.collect_images, daemon=True).start()

    def stop_collection(self):
        self.collecting = False
        self.collect_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_text.set(f"Not collecting")
        self.count_lbl.configure(text="0/100 images collected")

    def collect_images(self):
        word = self.word_var.get().replace(" ", "_")
        save_path = os.path.join(DATA_DIR, word)
        imgs_now = len(os.listdir(save_path))
        for i in range(imgs_now, NUM_IMAGES_PER_CLASS):
            if not self.collecting:
                break
            ret, frame = self.cap.read()
            if ret:
                filename = os.path.join(save_path, f"{i}.jpg")
                cv2.imwrite(filename, frame)
                self.counter = i + 1
                self.count_lbl.configure(text=f"{self.counter}/{NUM_IMAGES_PER_CLASS} images collected")
            else:
                break
            cv2.waitKey(70)
        self.collecting = False
        self.collect_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if self.counter >= NUM_IMAGES_PER_CLASS:
            self.status_text.set("Collection complete!")
        else:
            self.status_text.set("Stopped.")
        self.count_lbl.configure(text=f"{self.counter}/{NUM_IMAGES_PER_CLASS} images collected")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.after(15, self.update_video)

    def train_model(self):
        self.train_btn.configure(state="disabled")
        self.training_status.set("Extracting hand landmarks...")
        self.save_btn.configure(state="disabled")
        self.data.clear()
        self.labels.clear()
        self.training_status_lbl.configure(text="")
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True)
        all_ok = True
        for label_idx, word in enumerate(SIGN_NAMES):
            folder = os.path.join(DATA_DIR, word.replace(" ", "_"))
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y])
                        if len(landmarks) == 42:
                            self.data.append(landmarks)
                            self.labels.append(label_idx)
        hands.close()
        X = np.array(self.data)
        y = np.array(self.labels)
        if len(X) == 0:
            self.training_status.set("Error: No valid data found! (Check images.)")
            self.train_btn.configure(state="normal")
            return
        self.training_status.set("Training model...")
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=y
        )
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        val_acc = accuracy_score(y_val, val_pred)
        self.model = model
        self.training_status.set(f"Model trained! Validation accuracy:")
        self.accuracy_text.set(f"{val_acc*100:.2f}%")
        self.save_btn.configure(state="normal")
        self.train_btn.configure(state="normal")
        self.training_status_lbl.configure(text=f"Trained on {len(y)} samples.")

    def save_model(self):
        if self.model is None:
            self.training_status.set("No model to save!")
            return
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({'model': self.model, 'labels': SIGN_NAMES}, f)
        self.training_status.set(f"Model saved as {MODEL_PATH} ✔")
        messagebox.showinfo("Model Saved", f"Model saved as {MODEL_PATH}.")

    def __del__(self):
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = SignTrainerGUI()
    app.mainloop()
