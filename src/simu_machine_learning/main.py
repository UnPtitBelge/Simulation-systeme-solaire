from parser import DataParser
import numpy as np
from neural_network import Neural_network
import matplotlib.pyplot as plt


def generate_trajectory(x0, y0, vx0, vy0, n_steps, ax=0.0, ay=-0.1):
    """
    Génère une trajectoire simple avec accélération constante.
    x0, y0 : point de départ
    vx0, vy0 : vitesse initiale
    n_steps : nombre de pas
    ax, ay : accélérations (simule gravité/frottement)
    """
    traj = [(x0, y0)]
    x, y = x0, y0
    vx, vy = vx0, vy0

    for _ in range(n_steps):
        vx += ax
        vy += ay
        x += vx
        y += vy
        traj.append((x, y))

    return traj


if __name__ == "__main__":
    # Génération des trajectoires
    trajectories = []
    for _ in range(100):
        x0, y0 = np.random.uniform(0, 1), np.random.uniform(0, 1)
        vx0, vy0 = np.random.uniform(0.05, 0.2), np.random.uniform(0.05, 0.2)
        traj = generate_trajectory(x0, y0, vx0, vy0, n_steps=5, ay=-0.05)
        trajectories.append(traj)

    print("Trajectoires générées pour l'entraînement.")

    # Parsing des données
    X, Y = DataParser.prepare_dataset(trajectories)
    print("Dataset préparé :", X.shape, Y.shape)

    # Entraînement du réseau
    nn = Neural_network(input_dim=2, hidden_dim=16, learning_rate=0.01)
    epochs = 1000
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_vec, y_vec in zip(X, Y):
            pred = nn.forward(np.array(x_vec))
            epoch_loss += nn.MSE(pred, np.array(y_vec))
            nn.backward(np.array(x_vec), np.array(y_vec))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {epoch_loss / len(X):.6f}")

    x, y = trajectories[0][0]
    trajectory_pred = [(x, y)]
    for _ in range(len(trajectories[0])):
        dx, dy = nn.forward(np.array([x, y]))
        x += dx
        y += dy
        trajectory_pred.append((x, y))

    print("Trajectoire prédite :", trajectory_pred)

    # Plot
    plt.figure(figsize=(6, 6))
    # Quelques trajectoires réelles pour lisibilité
    for traj in trajectories[:30]:
        X_real, Y_real = zip(*traj)
        plt.plot(
            X_real,
            Y_real,
            "o-",
            alpha=0.3,
            label="Réalité" if traj == trajectories[0] else "",
        )
    # Trajectoire prédite
    X_pred, Y_pred = zip(*trajectory_pred)
    plt.plot(X_pred, Y_pred, "x--", color="red", label="Prédiction NN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Trajectoire réelle vs prédite")
    plt.show()
