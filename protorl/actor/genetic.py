from protorl.actor.base import Actor
import torch as T

# actor for genetic algorithm
class GeneticActor(Actor):
    def __init__(self, network, policy, tau=1.0, device=None):
        super().__init__(policy=policy, tau=tau, device=device)
        self.networks = [network]

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        outputs = self.networks[0](state)
        action = self.policy(outputs)
        return action
