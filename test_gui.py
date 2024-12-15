import tkinter as tk
from tkinter import ttk
import os

# Costanti per le directory dei modelli pre-addestrati
QL_MODELS_DIR = "qlearning_models"
SARSA_MODELS_DIR = "sarsa_models"


def get_model_files(agent_type):
    """
    Ottiene i file dei modelli pre-addestrati dalla directory corrispondente.
    :param agent_type: Tipo di agente ('qlearning' o 'sarsa').
    :return: Lista di file presenti nella directory.
    """
    directory = QL_MODELS_DIR if agent_type == "qlearning" else SARSA_MODELS_DIR
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def update_fields():
    """
    Mostra o nasconde i campi in base alle scelte dell'utente.
    """
    mode = mode_var.get()
    train_new = train_var.get()

    # Mostra o nasconde i campi per Agent vs Agent o Agent vs Player
    if mode == "agent_vs_agent":
        left_agent_frame.pack(fill="x", pady=5)
        right_agent_frame.pack(fill="x", pady=5)
        player_frame.pack_forget()
    elif mode == "agent_vs_player":
        left_agent_frame.pack(fill="x", pady=5)
        right_agent_frame.pack_forget()
        player_frame.pack(fill="x", pady=5)

    # Mostra o nasconde i campi per addestrare nuovi agenti
    update_agent_config(left_agent_type, left_agent_config_frame, train_new)
    if mode == "agent_vs_agent":
        update_agent_config(right_agent_type, right_agent_config_frame, train_new)


def update_agent_config(agent_type_dropdown, config_frame, train_new):
    """
    Aggiorna la configurazione per un agente specifico.
    :param agent_type_dropdown: Dropdown per selezionare il tipo di agente.
    :param config_frame: Frame che ospita la configurazione dell'agente.
    :param train_new: Se True, mostra i campi per configurare i parametri di addestramento.
    """
    for widget in config_frame.winfo_children():
        widget.destroy()

    if train_new:
        # Parametri di addestramento
        ttk.Label(config_frame, text="Epsilon Start:").pack(anchor="w")
        ttk.Entry(config_frame).pack(anchor="w")

        ttk.Label(config_frame, text="Epsilon End:").pack(anchor="w")
        ttk.Entry(config_frame).pack(anchor="w")

        ttk.Label(config_frame, text="Learning Rate:").pack(anchor="w")
        ttk.Entry(config_frame).pack(anchor="w")
    else:
        # Selezione modelli pre-addestrati
        ttk.Label(config_frame, text="Seleziona Modello:").pack(anchor="w")
        model_dropdown = ttk.Combobox(config_frame, values=get_model_files(agent_type_dropdown.get()))
        model_dropdown.pack(anchor="w")
        model_dropdown.set("")


# Inizializza la finestra principale
root = tk.Tk()
root.title("Configurazione Partita")
root.geometry("600x700")

# Modalità di gioco
mode_var = tk.StringVar(value="agent_vs_agent")
train_var = tk.BooleanVar(value=False)

# Sezione 1: Seleziona modalità
mode_frame = tk.LabelFrame(root, text="Modalità di Gioco", padx=10, pady=10)
mode_frame.pack(fill="x", padx=10, pady=10)

ttk.Radiobutton(mode_frame, text="Agent vs Agent", variable=mode_var, value="agent_vs_agent", command=update_fields).pack(anchor="w")
ttk.Radiobutton(mode_frame, text="Agent vs Player", variable=mode_var, value="agent_vs_player", command=update_fields).pack(anchor="w")

# Sezione 2: Addestrare o utilizzare modelli pre-addestrati
train_frame = tk.LabelFrame(root, text="Opzioni", padx=10, pady=10)
train_frame.pack(fill="x", padx=10, pady=10)

ttk.Checkbutton(train_frame, text="Addestra nuovi agenti", variable=train_var, command=update_fields).pack(anchor="w")

# Sezione 3: Configura agenti
agent_frame = tk.LabelFrame(root, text="Configurazione Agenti", padx=10, pady=10)
agent_frame.pack(fill="x", padx=10, pady=10)

# Frame per il lato sinistro (sempre visibile)
left_agent_frame = tk.Frame(agent_frame)
left_agent_frame.pack(side="top", fill="x", pady=5)  # Cambiato da pack() a side="top"
tk.Label(left_agent_frame, text="Agente Sinistra").pack(anchor="w")
left_agent_type = ttk.Combobox(left_agent_frame, values=["qlearning", "sarsa"])
left_agent_type.pack(anchor="w")
left_agent_type.set("qlearning")
left_agent_config_frame = tk.Frame(left_agent_frame)
left_agent_config_frame.pack(anchor="w", pady=5)

# Frame per il lato destro (solo in Agent vs Agent)
right_agent_frame = tk.Frame(agent_frame)
right_agent_frame.pack(side="top", fill="x", pady=5)  # Cambiato da pack() a side="top"
tk.Label(right_agent_frame, text="Agente Destra").pack(anchor="w")
right_agent_type = ttk.Combobox(right_agent_frame, values=["qlearning", "sarsa"])
right_agent_type.pack(anchor="w")
right_agent_type.set("qlearning")
right_agent_config_frame = tk.Frame(right_agent_frame)
right_agent_config_frame.pack(anchor="w", pady=5)

# Frame per il player (solo in Agent vs Player)
player_frame = tk.Frame(agent_frame)
tk.Label(player_frame, text="Player").pack(anchor="w")
ttk.Label(player_frame, text="Il player sarà controllato manualmente.").pack(anchor="w")

# Bottoni per avviare
ttk.Button(root, text="Avvia Partita", command=lambda: print("Partita Avviata")).pack(pady=10)

# Aggiorna campi iniziali
update_fields()

# Avvia GUI
root.mainloop()