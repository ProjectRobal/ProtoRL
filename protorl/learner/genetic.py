from protorl.learner.base import Learner
import numpy as np
import torch as T

# it will perform crossover and mutation 

class GeneticLearner(Learner):
    def __init__(self, mutation_probability,mutation_mean=0.0,mutation_std=0.01,
                 tau=1.0, gamma=0.99, lr=1e-4):
        super().__init__(gamma=gamma, tau=tau)
        self.mutation_probability=mutation_probability
        self.gauss = (mutation_mean,mutation_std)

    def update(self, transitions):
        pass
