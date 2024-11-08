import pandas as pd
import tkinter as tk
from tkinter import messagebox, Toplevel, Text

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
root.geometry("500x400")  # Fenstergröße (Breite x Höhe)

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
        # Nur relevante Spalten anzeigen
        result_df = filtered_df[['title', 'overview']]
        
        # Neues Fenster öffnen, um die Ergebnisse anzuzeigen
        result_window = Toplevel(root)
        result_window.title("Suchergebnisse")
        result_window.geometry("600x400")
        
        # Text-Widget zur Anzeige der Ergebnisse
        result_text = Text(result_window, wrap="word")
        result_text.insert(tk.END, result_df.to_string(index=False))
        result_text.pack(expand=True, fill="both")

# Label und Eingabefelder hinzufügen
schauspieler_label = tk.Label(root, text="Gib den Schauspieler ein:")
schauspieler_label.pack(pady=5)
schauspieler_entry = tk.Entry(root, width=40)
schauspieler_entry.pack(pady=5)

genre_label = tk.Label(root, text="Gib das Genre ein:")
genre_label.pack(pady=5)
genre_entry = tk.Entry(root, width=40)
genre_entry.pack(pady=5)

# Button hinzufügen
button = tk.Button(root, text="Suche starten", command=button_click)
button.pack(pady=20)

# Das Tkinter-Fenster starten und offen halten
root.mainloop()
