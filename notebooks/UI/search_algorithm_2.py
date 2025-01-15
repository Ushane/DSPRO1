import tkinter as tk
from tkinter import messagebox, ttk
import csv
import threading
import pygame
import cv2
from PIL import Image, ImageTk
import os
import datetime
import pandas as pd
import joblib as jl

# Modell laden
model_path = "C:/Users/Narul/OneDrive/Dokumente/Hochschule Luzern 2023/Semester 3/DSPRO1/movie_ratings_prediction.joblib"
model = jl.load(model_path)

# Erwartete Features für das Modell
required_features = ['Unnamed: 0','budget', 'popularity', 'revenue', 'Action', 'Adventure', 'Animation', 'Comedy', 
                     'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 
                     'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 
                     'Western', 'top_actors_count']

# Standardwerte für fehlende Features
DEFAULT_REVENUE = 11200000
DEFAULT_POPULARITY = 2.92

# Musik abspielen
def play_music():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(
            "C:/Users/Narul/OneDrive/Dokumente/Hochschule Luzern 2023/Semester 3/DSPRO1/DSPRO1/DSPRO1/pictures/Outdated Time.mp3"
        )
        pygame.mixer.music.play(-1)
    except pygame.error as e:
        print(f"Fehler beim Starten der Musik: {e}")

# Video im Hintergrund abspielen
def play_video(video_label, width, height):
    cap = cv2.VideoCapture(
        "C:/Users/Narul/OneDrive/Dokumente/Hochschule Luzern 2023/Semester 3/DSPRO1/DSPRO1/DSPRO1/pictures/animated_backround.mp4"
    )

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.after(10, update_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            update_frame()

    update_frame()

# Suchfunktion
def search_logic(schauspieler, genre):
    if not schauspieler and not genre:
        messagebox.showinfo("Hinweis", "Bitte mindestens ein Feld ausfüllen.")
        return
    messagebox.showinfo("Erfolg", f"Suche nach Schauspieler: {schauspieler} und Genre: {genre}")

# Speicherlogik
def save_logic(budget, top_schauspieler, genre_vars):
    try:
        budget = int(budget)
        top_schauspieler = int(top_schauspieler)
    except ValueError:
        messagebox.showerror("Fehler", "Budget und Top-Schauspieler müssen Zahlen sein.")
        return

    genres_selected = {genre: var.get() for genre, var in genre_vars.items()}

    # Sicherstellen, dass "top_actor_count" (kein "s") als Name verwendet wird
    data = {
        "budget": budget,
        "popularity": DEFAULT_POPULARITY,
        "revenue": DEFAULT_REVENUE,
        **genres_selected,
        "top_actor_count": top_schauspieler  # Korrektur des Namens
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"movie_data_{timestamp}.csv"
    file_path = os.path.join(
        "C:/Users/Narul/OneDrive/Dokumente/Hochschule Luzern 2023/Semester 3/DSPRO1/DSPRO1/data/UI_input_data",
        file_name
    )

    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    messagebox.showinfo("Erfolg", f"Daten gespeichert in Datei: {file_name}")
    predict_from_csv(file_path)


# Vorhersagen
def predict_from_csv(file_path):
    try:
        # CSV-Datei einlesen
        test_data = pd.read_csv(file_path)

        # Entferne zusätzliche Spalten wie "Unnamed: 0"
        #test_data = test_data.drop(columns=['Unnamed: 0'])

        # Sicherstellen, dass alle erwarteten Features vorhanden sind
        missing_features = [feature for feature in required_features if feature not in test_data.columns]
        for feature in missing_features:
            # Fehlende Spalten mit Standardwerten auffüllen
            test_data[feature] = 0  # Oder Standardwerte wie DEFAULT_REVENUE etc.
        
        

        # Sicherstellen, dass die Spalten in der richtigen Reihenfolge sind
        X_test = test_data[required_features]

        # Vorhersagen mit dem Modell durchführen
        predictions = model.predict(X_test)

        # Ergebnisse anzeigen
        messagebox.showinfo("Vorhersage abgeschlossen", f"Die vorhergesagten Bewertungen sind: {predictions}")
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler bei der Vorhersage: {e}")




# Hauptfenster erstellen
root = tk.Tk()
root.title("Hauptmenü")
root.geometry("600x700")

# Hintergrundvideo
video_label = tk.Label(root)
video_label.place(x=0, y=0, relwidth=1, relheight=1)
thr_video = threading.Thread(target=play_video, args=(video_label, 600, 700), daemon=True)
thr_video.start()

# Musik starten
thr_music = threading.Thread(target=play_music, daemon=True)
thr_music.start()

# Tabs erstellen
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Tab 1: Suchfunktion
search_tab = tk.Frame(notebook, bg="#000000")
notebook.add(search_tab, text="Suchfunktion")

tk.Label(search_tab, text="Schauspieler eingeben:", bg="#000000", fg="white", font=("Helvetica", 12)).place(relx=0.5, rely=0.4, anchor="center")
schauspieler_entry = tk.Entry(search_tab, font=("Helvetica", 12))
schauspieler_entry.place(relx=0.5, rely=0.45, anchor="center")

tk.Label(search_tab, text="Genre eingeben:", bg="#000000", fg="white", font=("Helvetica", 12)).place(relx=0.5, rely=0.5, anchor="center")
genre_entry = tk.Entry(search_tab, font=("Helvetica", 12))
genre_entry.place(relx=0.5, rely=0.55, anchor="center")

search_button = tk.Button(search_tab, text="Suche starten", command=lambda: search_logic(schauspieler_entry.get(), genre_entry.get()), font=("Helvetica", 12), bg="#4CAF50", fg="white")
search_button.place(relx=0.5, rely=0.65, anchor="center")

# Tab 2: Speicherfunktion
save_tab = tk.Frame(notebook, bg="#f0f4f7")
notebook.add(save_tab, text="Speicherfunktion")

tk.Label(save_tab, text="Budget eingeben:", bg="#f0f4f7", fg="#333333", font=("Helvetica", 14)).place(relx=0.2, rely=0.2)
budget_entry = tk.Entry(save_tab, font=("Helvetica", 14))
budget_entry.place(relx=0.5, rely=0.2)

tk.Label(save_tab, text="Top-Schauspieler eingeben:", bg="#f0f4f7", fg="#333333", font=("Helvetica", 14)).place(relx=0.2, rely=0.3)
top_schauspieler_entry = tk.Entry(save_tab, font=("Helvetica", 14))
top_schauspieler_entry.place(relx=0.5, rely=0.3)

genre_frame = tk.LabelFrame(save_tab, text="Genres", font=("Helvetica", 14), padx=10, pady=10, bg="#f0f4f7")
genre_frame.place(relx=0.2, rely=0.4, relwidth=0.6, relheight=0.4)

genre_vars = {}
genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", 
          "Foreign", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", 
          "Thriller", "War", "Western"]
for i, genre in enumerate(genres):
    var = tk.IntVar()
    tk.Checkbutton(genre_frame, text=genre, variable=var, font=("Helvetica", 12), bg="#f0f4f7").grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=5)
    genre_vars[genre] = var

save_button = tk.Button(save_tab, text="Speichern", command=lambda: save_logic(budget_entry.get(), top_schauspieler_entry.get(), genre_vars), font=("Helvetica", 14), bg="#4CAF50", fg="white")
save_button.place(relx=0.5, rely=0.85, anchor="center")

root.mainloop()
