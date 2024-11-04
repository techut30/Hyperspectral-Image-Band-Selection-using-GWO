# Gray Wolf Optimization Algorithm for Band Selection
import numpy as np

class GrayWolfOptimizer:
    def __init__(self, objective_function, dim, lb, ub, num_wolves, max_iter):
        self.objective_function = objective_function
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.num_wolves = num_wolves
        self.max_iter = max_iter

        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")

        self.beta_pos = np.zeros(dim)
        self.beta_score = float("inf")

        self.delta_pos = np.zeros(dim)
        self.delta_score = float("inf")

        self.positions = np.random.uniform(low=self.lb, high=self.ub, size=(self.num_wolves, self.dim))

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                fitness = self.objective_function(self.positions[i, :])

                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()

                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i, :].copy()

                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = fitness
                    self.beta_pos = self.positions[i, :].copy()

                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i, :].copy()

            a = 2 - iter * (2 / self.max_iter)

            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.random(), np.random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.random(), np.random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.random(), np.random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                    self.positions[i, j] = (X1 + X2 + X3) / 3
                    self.positions[i, j] = np.clip(self.positions[i, j], self.lb, self.ub)

        return self.alpha_pos, self.alpha_score
