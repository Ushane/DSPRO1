
import pandas as pd
import tkinter as tk
from tkinter import messagebox


pd.set_option('display.max_columns', None)  # Zeige alle Spalten an
df = pd.read_csv("data/archive/raw/movies_metadata.csv")
#print(df.head(10))

print(df.columns)




# Hauptfenster erstellen

"""
root = tk.Tk()
root.title("Einfaches User Interface")
root.geometry("300x200")  # Fenstergröße (Breite x Höhe)

# Funktion, die aufgerufen wird, wenn der Button gedrückt wird
def button_click():
    user_input = entry.get()
    
    messagebox.showinfo("Eingabe", f"Du hast eingegeben: {user_input}")

# Label hinzufügen
label = tk.Label(root, text="Gib etwas ein:")
label.pack(pady=10)

# Eingabefeld (Entry) hinzufügen
entry = tk.Entry(root, width=25)
entry.pack(pady=10)

# Button hinzufügen
button = tk.Button(root, text="Klick mich!", command=button_click)
button.pack(pady=10)

# Das Tkinter-Fenster starten und offen halten
root.mainloop() """