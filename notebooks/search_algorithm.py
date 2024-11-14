import pandas as pd
import tkinter as tk
from tkinter import messagebox, Toplevel, Text, ttk, font as tkfont
import cv2
from PIL import Image, ImageTk
import threading
import pygame  # Modul für die Musikwiedergabe

# Musikdatei initialisieren und starten
def play_music():
    pygame.mixer.init()
    pygame.mixer.music.load("C:/Users/Narul/OneDrive/Dokumente/Hochschule Luzern 2023/Semester 3/DSPRO1/DSPRO1/DSPRO1/pictures/Outdated Time.mp3"
)  # Pfad zur Musikdatei
    pygame.mixer.music.play(-1)  # Dauerschleife (-1 bedeutet, dass es endlos wiederholt wird)

# Datenbank laden
pd.set_option('display.max_columns', None)  # Zeige alle Spalten an
movies_df = pd.read_csv("data/archive/raw/movies_metadata.csv")
credits_df = pd.read_csv("data/archive/raw/credits.csv")

# Konvertiere die 'id'-Spalten in beiden DataFrames zu Strings, um sie zusammenzuführen
movies_df['id'] = movies_df['id'].astype(str)
credits_df['id'] = credits_df['id'].astype(str)

# Hauptfenster erstellen
root = tk.Tk()
root.title("Film Suchinterface")
root.geometry("500x600")  # Fenstergröße entsprechend dem Video
root.configure(bg="#000000")  # Schwarzer Hintergrund

# Video-Label hinzufügen
video_label = tk.Label(root)
video_label.place(x=0, y=0, relwidth=1, relheight=1)  # Fullscreen im Hintergrund

# Funktion, um das Video als Hintergrund abzuspielen
def play_video():
    cap = cv2.VideoCapture("C:/Users/Narul/OneDrive/Dokumente/Hochschule Luzern 2023/Semester 3/DSPRO1/DSPRO1/DSPRO1/pictures/animated_backround.mp4")

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (500, 600))  # Anpassen der Fenstergröße
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.after(10, update_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Video zurückspulen
            update_frame()

    update_frame()

# Stil für Widgets festlegen
style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 12), background="#000000", foreground="white")
style.configure("TEntry", font=("Helvetica", 12), background="#ffffff")
style.configure("Custom.TButton", font=("Helvetica", 12), background="#000000", foreground="white", borderwidth=0)
style.map("Custom.TButton", background=[('active', '#333333')])  # Hintergrundfarbe beim Hover

# Funktion, die aufgerufen wird, wenn der Button gedrückt wird
def button_click():
    schauspieler_input = schauspieler_entry.get().strip().lower()
    genre_input = genre_entry.get().strip().lower()

    # Filterlogik basierend auf Benutzereingaben
    merged_df = pd.merge(movies_df, credits_df, on='id')
    filtered_df = merged_df.copy()

    if schauspieler_input:
        filtered_df = filtered_df[filtered_df['cast'].astype(str).str.lower().str.contains(schauspieler_input)]

    if genre_input:
        filtered_df = filtered_df[filtered_df['genres'].astype(str).str.lower().str.contains(genre_input)]

    # Prüfen, ob es Ergebnisse gibt
    if filtered_df.empty:
        messagebox.showinfo("Ergebnis", "Keine Ergebnisse gefunden.")
    else:
        # Neues Fenster öffnen, um die Ergebnisse anzuzeigen
        result_window = Toplevel(root)
        result_window.title("Suchergebnisse")
        result_window.geometry("500x400")
        result_window.configure(bg="#f0f8ff")
        
        # Text-Widget zur Anzeige der Ergebnisse mit Formatierung
        result_text = Text(result_window, wrap="word", font=("Helvetica", 10), bg="#ffffff", fg="#000000")
        result_text.pack(expand=True, fill="both", padx=10, pady=10)

        # Font-Definitionen
        bold_font = tkfont.Font(result_text, result_text.cget("font"))
        bold_font.configure(weight="bold")
        normal_font = tkfont.Font(result_text, result_text.cget("font"))
        normal_font.configure(size=10)

        # Ergebnisse formatieren
        for _, row in filtered_df.iterrows():
            title = row['title']
            overview = row['overview']
            if pd.notna(title) and pd.notna(overview):
                result_text.insert(tk.END, f"{title}\n", "bold")
                result_text.insert(tk.END, f"{overview}\n\n", "normal")

        # Tags für Formatierungen hinzufügen
        result_text.tag_configure("bold", font=bold_font)
        result_text.tag_configure("normal", font=normal_font)

# Widgets für die Eingabe ohne Rahmenhintergrund hinzufügen
schauspieler_label = ttk.Label(root, text="Gib den Schauspieler ein:", background="#000000", foreground="white")
schauspieler_label.place(relx=0.5, rely=0.55, anchor="center")
schauspieler_entry = ttk.Entry(root, width=40)
schauspieler_entry.place(relx=0.5, rely=0.6, anchor="center")

genre_label = ttk.Label(root, text="Gib das Genre ein:", background="#000000", foreground="white")
genre_label.place(relx=0.5, rely=0.65, anchor="center")
genre_entry = ttk.Entry(root, width=40)
genre_entry.place(relx=0.5, rely=0.7, anchor="center")

# Button hinzufügen
button = ttk.Button(root, text="Suche starten", style="Custom.TButton", command=button_click)
button.place(relx=0.5, rely=0.75, anchor="center")

# Video-Thread starten
video_thread = threading.Thread(target=play_video, daemon=True)
video_thread.start()

# Musik-Thread starten
music_thread = threading.Thread(target=play_music, daemon=True)
music_thread.start()

# Das Tkinter-Fenster starten und offen halten
root.mainloop()
