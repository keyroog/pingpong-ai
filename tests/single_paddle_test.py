from environments.single_paddle_env import SinglePaddleEnv

# Crea l'ambiente
env = SinglePaddleEnv()

# Reset dell'ambiente
state = env.reset()

done = False
total_reward = 0

while not done:
    # Azione casuale (per test, da sostituire con un agente RL o input utente)
    action = env.action_space.sample()
    
    # Passo nell'ambiente
    state, reward, done, info = env.step(action)
    total_reward += reward

    # Stampa stato e punteggio
    print(f"Stato: {state}, Ricompensa: {reward}, Punteggio: {info['score']}")

print(f"Gioco terminato. Punteggio finale: {info['score']}, Ricompensa totale: {total_reward}")