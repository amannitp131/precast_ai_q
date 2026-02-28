import numpy as np


class QPSO:
    def __init__(
        self,
        fitness_function,
        dim,
        bounds,
        particles=30,
        iterations=50,
        beta_max=1.0,
        beta_min=0.5,
        seed=None,
    ):
        self.fitness = fitness_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.particles = particles
        self.iterations = iterations
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.seed = seed

    def optimize(self):
        rng = np.random.default_rng(self.seed)

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        x = rng.uniform(low=low, high=high, size=(self.particles, self.dim))

        pbest = x.copy()
        pbest_scores = np.array([self.fitness(p) for p in pbest])

        gbest_index = int(np.argmin(pbest_scores))
        gbest = pbest[gbest_index].copy()
        gbest_score = float(pbest_scores[gbest_index])

        for iteration in range(self.iterations):
            beta = self.beta_max - (self.beta_max - self.beta_min) * (iteration / max(1, self.iterations - 1))
            mbest = np.mean(pbest, axis=0)

            for i in range(self.particles):
                phi = rng.random(self.dim)
                p = phi * pbest[i] + (1 - phi) * gbest

                u = np.clip(rng.random(self.dim), 1e-12, 1 - 1e-12)
                direction = np.where(rng.random(self.dim) < 0.5, -1.0, 1.0)

                candidate = p + direction * beta * np.abs(mbest - x[i]) * np.log(1 / u)
                candidate = np.clip(candidate, low, high)

                candidate_score = float(self.fitness(candidate))
                x[i] = candidate

                if candidate_score < pbest_scores[i]:
                    pbest[i] = candidate
                    pbest_scores[i] = candidate_score

                    if candidate_score < gbest_score:
                        gbest = candidate.copy()
                        gbest_score = candidate_score

        return gbest, gbest_score
