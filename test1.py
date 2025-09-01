import customtkinter as ctk
from tkinter import scrolledtext
import cv2
import pickle
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

MODEL_PATH = "model.p"

class SignRecognizerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language Recognition - CustomTkinter Edition")
        self.geometry("920x600")
        self.resizable(False, False)
        if not os.path.exists(MODEL_PATH):
            ctk.CTkMessagebox(title="Model not found", message="Trained model.p not found!", icon="cancel")
            self.destroy()
        model_dict = pickle.load(open(MODEL_PATH, "rb"))
        self.model = model_dict['model']
        self.labels_list = model_dict['labels'] if 'labels' in model_dict else ["Class"+str(i) for i in range(10)]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None
        self.running = False
        self.cooldown_time = 1.0
        self.last_pred_time = 0
        self.word_buffer = ""
        mainframe = ctk.CTkFrame(self, fg_color=("white", "#23222e"), corner_radius=18)
        mainframe.pack(padx=23, pady=15, fill="both", expand=True)
        side = ctk.CTkFrame(mainframe, width=450, height=340, corner_radius=18)
        side.pack(side="left", padx=27, pady=35)
        self.video_label = ctk.CTkLabel(side, text="", width=430, height=320, corner_radius=10)
        self.video_label.pack(pady=8, padx=8)
        btns = ctk.CTkFrame(side, fg_color="transparent")
        btns.pack(pady=(14, 5))
        self.start_btn = ctk.CTkButton(btns, text="Start Recognition", command=self.start_recognition, width=155)
        self.start_btn.pack(side="left", padx=9)
        self.stop_btn = ctk.CTkButton(btns, text="Stop", command=self.stop_recognition, width=80, state="disabled")
        self.stop_btn.pack(side="left", padx=9)
        self.clear_btn = ctk.CTkButton(btns, text="Clear Output", command=self.clear_output, width=110)
        self.clear_btn.pack(side="left", padx=9)
        predframe = ctk.CTkFrame(mainframe, width=480, height=340, corner_radius=18)
        predframe.pack(side="right", padx=20, pady=35, fill="both", expand=True)
        ctk.CTkLabel(predframe, text="Current Prediction:", font=("Segoe UI SemiBold", 22)).pack(pady=(38,7))
        self.pred_result = ctk.CTkLabel(predframe, text="-", font=("Segoe UI Black", 52), fg_color="#e1ecf7", text_color="#183c6b", width=245, height=68, corner_radius=14)
        self.pred_result.pack()
        ctk.CTkLabel(predframe, text="Predicted Word Buffer:", font=("Segoe UI", 17)).pack(pady=(30, 1))
        self.word_entry = ctk.CTkEntry(predframe, font=("Arial", 18), width=320, height=36, corner_radius=9, state="readonly", text_color="#228c39")
        self.word_entry.pack(pady=(3, 14))
        self.word_entry.insert(0, "")
        ctk.CTkLabel(predframe, text="History:", font=("Arial", 15, "bold")).pack()
        self.scroll_pred = ctk.CTkScrollableFrame(predframe, width=410, height=100, orientation="horizontal", fg_color="#f4f7fe")
        self.scroll_pred.pack(pady=(1, 8))
        self.prediction_history = []

    def start_recognition(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.word_buffer = ""
        self.prediction_history.clear()
        self.update_history()
        self.pred_result.configure(text="-")
        self.word_entry.configure(state="normal")
        self.word_entry.delete(0, "end")
        self.word_entry.configure(state="readonly")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def stop_recognition(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.pred_result.configure(text="-")

    def clear_output(self):
        self.word_buffer = ""
        self.prediction_history.clear()
        self.pred_result.configure(text="-")
        self.word_entry.configure(state="normal")
        self.word_entry.delete(0, "end")
        self.word_entry.configure(state="readonly")
        self.update_history()

    def camera_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(frame_rgb)

            predicted_word = "-"
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])
                    current_time = time.time()
                    if len(landmarks) == 42 and (current_time - self.last_pred_time) >= self.cooldown_time:
                        prediction = self.model.predict([np.array(landmarks)])
                        predicted_word = self.labels_list[int(prediction[0])]
                        self.last_pred_time = current_time
                        self.word_buffer += predicted_word + " "
                        self.prediction_history.append(predicted_word)
                        if len(self.prediction_history) > 18:
                            self.prediction_history.pop(0)
                        self.pred_result.configure(text=predicted_word)
                        self.word_entry.configure(state="normal")
                        self.word_entry.delete(0, "end")
                        self.word_entry.insert(0, self.word_buffer)
                        self.word_entry.configure(state="readonly")
                        self.update_history()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.imgtk = imgtk
            self.after(10, lambda: None)  
        if self.cap:
            self.cap.release()

    def update_history(self):
        for widget in self.scroll_pred.winfo_children():
            widget.destroy()
        for w in self.prediction_history:
            ctk.CTkLabel(self.scroll_pred, text=w, fg_color="#adc4ed", text_color="#2e3977", width=63, height=30, corner_radius=11, font=("Arial", 17, "bold")).pack(side="left", padx=6, pady=3)

if __name__ == "__main__":
    app = SignRecognizerGUI()
    app.mainloop()
