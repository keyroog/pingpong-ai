import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import os
from tkinter import messagebox
from utils.parameters import Q_Parameters, SARSA_Parameters  # Import default parameters
from main import start_main

# Directory constants for pre-trained models
QL_MODELS_DIR = "models/qlearning_models"
SARSA_MODELS_DIR = "models/sarsa_models"

def launch_game_config_gui():
    """
    Function to launch the Game Configuration GUI.
    """
    def get_model_files(agent_type, side):
        """
        Fetch pre-trained model files for the given agent type and side.
        :param agent_type: The type of agent (e.g., 'qlearning', 'sarsa').
        :param side: The side of the agent ('left' or 'right').
        :return: List of available models for the given agent type and side.
        """
        directory = QL_MODELS_DIR if agent_type == "qlearning" else SARSA_MODELS_DIR
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith(f"_{side}.pkl")]


    def update_fields():
        """Show or hide fields based on user choices and resize the window dynamically."""
        mode = mode_var.get()
        train_new = train_var.get()

        # Show/hide agent frames
        if mode == "agent_vs_agent":
            left_agent_frame.pack(fill=X, pady=5)
            right_agent_frame.pack(fill=X, pady=5)
            player_frame.pack_forget()
        elif mode == "agent_vs_player":
            left_agent_frame.pack(fill=X, pady=5)
            right_agent_frame.pack_forget()
            player_frame.pack(fill=X, pady=5)

        # Show or hide the episodes field
        if train_new:
            episodes_frame.pack(fill=X, pady=5)  # Show episodes field
        else:
            episodes_frame.pack_forget()  # Hide episodes field

        # Update agent configuration based on training option
        update_agent_config(left_agent_type, left_agent_config_frame, train_new, "left")
        if mode == "agent_vs_agent":
            update_agent_config(right_agent_type, right_agent_config_frame, train_new, "right")

        # Dynamically resize the window
        root.update_idletasks()
        root.geometry(f"{root.winfo_reqwidth()}x{root.winfo_reqheight()}")


    def update_agent_config(agent_type_dropdown, config_frame, train_new, side):
        """
        Update the agent configuration frame.
        :param agent_type_dropdown: The dropdown widget for selecting the agent type.
        :param config_frame: The frame to update with configuration options.
        :param train_new: Boolean indicating if new agents are being trained.
        :param side: The side of the agent ('left' or 'right').
        """
        for widget in config_frame.winfo_children():
            widget.destroy()

        if train_new:
            # Add training parameters
            params = Q_Parameters if agent_type_dropdown.get() == "qlearning" else SARSA_Parameters
            for param, default_value in params.items():
                ttk.Label(config_frame, text=f"{param}:", font=("Helvetica", 10)).pack(anchor="w")
                entry = ttk.Entry(config_frame)
                entry.insert(0, str(default_value))  # Prepopulate with default value
                entry.pack(anchor="w", pady=2)
                training_params[agent_type_dropdown.get()][param] = entry  # Store the entry widget for retrieval
        else:
            # Add pre-trained model selection
            ttk.Label(config_frame, text="Seleziona Modello:", font=("Helvetica", 10)).pack(anchor="w")
            model_dropdown = ttk.Combobox(
                config_frame,
                values=get_model_files(agent_type_dropdown.get(), side),
                state="readonly"
            )
            model_dropdown.pack(anchor="w", pady=2)
            model_dropdown.set("")


    def validate_and_start():
        """Validate fields before starting the game and pass the configuration."""
        train_new = train_var.get()
        config = {"mode": mode_var.get(), "train_new": train_new}

        # Validate episodes
        if train_new:
            episodes = episodes_entry.get()
            if not episodes.isdigit() or int(episodes) <= 0:
                messagebox.showerror("Errore", "Inserisci un numero valido di episodi.")
                return
            config["episodes"] = int(episodes)

            # Collect training parameters
            left_params = {param: widget.get() for param, widget in training_params["qlearning"].items()}
            right_params = {param: widget.get() for param, widget in training_params["sarsa"].items()}
            config["left_agent_params"] = {
                "epsilon_start": float(left_params.get("epsilon_start", 1.0)),
                "epsilon_end": float(left_params.get("epsilon_end", 0.1)),
                "epsilon_decay": int(left_params.get("epsilon_decay", 200)),
                "alpha": float(left_params.get("alpha", 0.1)),
                "alpha_end": float(left_params.get("alpha_end", 0.01)),
                "alpha_decay": float(left_params.get("alpha_decay", 0.99)),
                "gamma": float(left_params.get("gamma", 0.99)),
            }

            config["right_agent_params"] = {
                "epsilon_start": float(right_params.get("epsilon_start", 1.0)),
                "epsilon_end": float(right_params.get("epsilon_end", 0.1)),
                "epsilon_decay": int(right_params.get("epsilon_decay", 200)),
                "alpha": float(right_params.get("alpha", 0.1)),
                "alpha_end": float(right_params.get("alpha_end", 0.01)),
                "alpha_decay": float(right_params.get("alpha_decay", 0.99)),
                "gamma": float(right_params.get("gamma", 0.99)),
            }
        else:
            # Validate model selection for pre-trained agents
            left_model = left_agent_config_frame.winfo_children()[1].get()
            if not left_model:
                messagebox.showerror("Errore", "Seleziona un modello per l'Agente Sinistra.")
                return
            config["left_model"] = left_model

            if mode_var.get() == "agent_vs_agent":
                right_model = right_agent_config_frame.winfo_children()[1].get()
                if not right_model:
                    messagebox.showerror("Errore", "Seleziona un modello per l'Agente Destra.")
                    return
                config["right_model"] = right_model

        # Collect agent types
        config["left_agent_type"] = left_agent_type.get()
        if mode_var.get() == "agent_vs_agent":
            config["right_agent_type"] = right_agent_type.get()

        
        # Close GUI
        root.destroy()

        # Call the main function with the collected configuration
        start_main(config)

    # Initialize the GUI
    root = ttk.Window(themename="superhero")  # Use ttkbootstrap themes
    root.title("Configurazione Partita")
    root.geometry("600x700")

    # Variables
    mode_var = ttk.StringVar(value="agent_vs_agent")
    train_var = ttk.BooleanVar(value=False)
    training_params = {"qlearning": {}, "sarsa": {}}  # Store references to training parameter entries

    # Game mode section
    mode_frame = ttk.Labelframe(root, text="Modalità di Gioco", padding=10)
    mode_frame.pack(fill=X, padx=10, pady=10)
    ttk.Radiobutton(mode_frame, text="Agent vs Agent", variable=mode_var, value="agent_vs_agent", command=update_fields).pack(anchor="w")
    ttk.Radiobutton(mode_frame, text="Agent vs Player", variable=mode_var, value="agent_vs_player", command=update_fields).pack(anchor="w")

    # Training options
    train_frame = ttk.Labelframe(root, text="Opzioni", padding=10)
    train_frame.pack(fill=X, padx=10, pady=10)
    ttk.Checkbutton(train_frame, text="Addestra nuovi agenti", variable=train_var, command=update_fields).pack(anchor="w")

    # Episodes configuration (top of agent frame)
    episodes_frame = ttk.Frame(root, padding=10)
    ttk.Label(episodes_frame, text="Numero di Episodi:").pack(side="left")
    episodes_entry = ttk.Entry(episodes_frame)
    episodes_entry.insert(0, "25000")  # Default episodes
    episodes_entry.pack(side="left")

    # Agent configuration
    agent_frame = ttk.Labelframe(root, text="Configurazione Agenti", padding=10)
    agent_frame.pack(fill=X, padx=10, pady=10)

    agents_container = ttk.Frame(agent_frame)
    agents_container.pack(fill=X, pady=5)

    # Left Agent
    left_agent_frame = ttk.Frame(agents_container, padding=10)
    left_agent_frame.pack(side="left", expand=True, fill=BOTH)
    ttk.Label(left_agent_frame, text="Agente Sinistra").pack(anchor="w")
    left_agent_type = ttk.Combobox(left_agent_frame, values=["qlearning", "sarsa"], state="readonly")
    left_agent_type.pack(anchor="w")
    left_agent_type.set("qlearning")
    left_agent_config_frame = ttk.Frame(left_agent_frame)
    left_agent_config_frame.pack(anchor="w", pady=5)

    # Bind event to reload parameters for left agent when changing type
    left_agent_type.bind("<<ComboboxSelected>>", lambda e: update_agent_config(left_agent_type, left_agent_config_frame, train_var.get(), "left"))

    # Right Agent
    right_agent_frame = ttk.Frame(agents_container, padding=10)
    right_agent_frame.pack(side="left", expand=True, fill=BOTH)
    ttk.Label(right_agent_frame, text="Agente Destra").pack(anchor="w")
    right_agent_type = ttk.Combobox(right_agent_frame, values=["qlearning", "sarsa"], state="readonly")
    right_agent_type.pack(anchor="w")
    right_agent_type.set("sarsa")
    right_agent_config_frame = ttk.Frame(right_agent_frame)
    right_agent_config_frame.pack(anchor="w", pady=5)

    # Bind event to reload parameters for right agent when changing type
    right_agent_type.bind("<<ComboboxSelected>>", lambda e: update_agent_config(right_agent_type, right_agent_config_frame, train_var.get(), "right"))

    # Player Configuration
    player_frame = ttk.Frame(agent_frame, padding=10)
    ttk.Label(player_frame, text="Player").pack(anchor="w")
    ttk.Label(player_frame, text="Il player sarà controllato manualmente.").pack(anchor="w")

    # Start Button
    ttk.Button(root, text="Avvia Partita", bootstyle=SUCCESS, command=validate_and_start).pack(pady=10)

    # Initialize fields
    update_fields()

    # Start GUI loop
    root.mainloop()

