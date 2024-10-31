from protorl.actor.base import Actor
import torch as T

import os

# actor for genetic algorithm
class GeneticActor(Actor):
    
    BASE_GENETIC_NAME = "genetic_base"
    HEAD_GENETIC_NAME = "genetic_head"
    
    def __init__(self, network,population_size, policy, tau=1.0, device=None):
        super().__init__(policy=policy, tau=tau, device=device)
        
        self.network = network
        
        # network reward and id
        self.population = [[0,i] for i in range(population_size)]
        
        self.population_size = population_size
        
        self.variant_id = 0
        
        self.shuttle()
        
    
    # return true if population is ready for crossover
    def shuttle(self):
        
        ret = False
        
        checkpoint_dir = self.network[0].checkpoint_dir
        
        base_name = self.BASE_GENETIC_NAME + "_{}".format(self.variant_id)
        head_name = self.HEAD_GENETIC_NAME + "_{}".format(self.variant_id)
        
        base_path = os.path.join(checkpoint_dir,base_name)
        head_path = os.path.join(checkpoint_dir,head_name)
        
        parent_base_path = os.path.join(checkpoint_dir,self.BASE_GENETIC_NAME)
        parent_head_path = os.path.join(checkpoint_dir,self.HEAD_GENETIC_NAME)
        
        os.rename(base_path,parent_base_path)
        os.rename(head_path,parent_head_path)
                
        for net in self.network:
            net.load_checkpoint()
        
        os.rename(parent_base_path,base_path)
        os.rename(parent_head_path,head_path)
        
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
        pass
        # self.network.save_checkpoint()

    def load_models(self):
        pass
        # self.network.load_checkpoint()
    
    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        outputs = self.network(state)
        action = self.policy(outputs)[0]
        return action
