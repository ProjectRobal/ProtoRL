from protorl.actor.base import Actor
import torch as T

# actor for genetic algorithm
class GeneticActor(Actor):
    def __init__(self, networks, policy, tau=1.0, device=None):
        super().__init__(policy=policy, tau=tau, device=device)
                
        self.population = [[0.0,net] for net in networks]
        
        self.variant_id = 0
        
    
    # return true if population is ready for crossover
    def shuttle(self):
        
        ret = False
        
        self.variant_id +=1
        
        ret = self.variant_id == len(self.population)
        
        self.variant_id = self.variant_id % len(self.population)
                        
        return ret
    
    # reset rewards
    def reset(self):
        for var in self.population:
            var[0] = 0.0
    
    def giveReward(self,reward):
        self.population[self.variant_id][0] += reward
    
    def givePopulation(self):
        return self.population
    
    def save_models(self):
        for (reward,network) in self.population:
            for x in network:
                x.save_checkpoint()

    def load_models(self):
        for (reward,network) in self.population:
            for x in network:
                x.load_checkpoint()
    
    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        outputs = self.population[self.variant_id][1](state)
        action = self.policy(outputs)[0]
        return action
