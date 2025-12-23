import numpy as np

class DataParser:
    def parse_data(self, experience: list[tuple]) -> tuple[list, list]:
        """
        Parse the experience data into input-output pairs for machine learning.

        Args:
            experience (list): A list of tuples representing the experience data, where each tuple contains input and output values.

        Returns:
            tuple: A tuple containing two lists, X and Y. X is a list of input pairs, and Y is a list of output differences.
        """
        X, Y = [], []
        for i in range(len(experience) - 1):
            x_t, y_t = experience[i]
            x_tp1, y_tp1 = experience[i + 1]

            X.append([x_t, y_t])
            Y.append([x_tp1 - x_t, y_tp1 - y_t])
        return X, Y
    
    @staticmethod
    def prepare_dataset(trajectories):
        """
        Transforme une liste de trajectoires en X et Y pour le réseau.
        Chaque trajectoire : liste de tuples (x, y)
        X : positions
        Y : Δx, Δy
        """
        X_all, Y_all = [], []
        parser = DataParser()
        for traj in trajectories:
            X, Y = parser.parse_data(traj)
            X_all.extend(X)
            Y_all.extend(Y)
        return np.array(X_all), np.array(Y_all)