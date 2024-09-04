from protorl.actor.base import Actor
import torch as T
import numpy as np

from random import choice

# actor for genetic algorithm
class ESPActor(Actor):
        
    def __init__(self, base_network, networks, policy,trail_number=10, tau=1.0, device=None):
        super().__init__(policy=policy, tau=tau, device=device)
        
        self.population_size = len(networks)
        
        self.trail_number = trail_number
        
        self.base_network = base_network
        
        self.population = networks   
        
        self.crossover_counters = None
        
        self.count_populations()
        
        self.add_metadata_to_network()
        
        # we will add to each row additional rows that will keep track of cummulative rewards and how much neuron was choosen           
        # let's util function generate additional columns
        
    def count_populations(self):
        population_count = 0
        for tensor in self.base_network.parameters():
            if len(tensor.shape) <= 1:
                continue
            
            population_count += tensor.shape[0]
            
        self.crossover_counters = np.zeros(population_count)   
            
        
    def add_metadata_to_network(self):
                
        for net in self.population:
            for tensor in net.parameters():
                detached = tensor.detach()
                
                if len(tensor.shape) <= 1:
                    continue
                
                height = detached.shape[0]
                                
                rewards = T.zeros(height).reshape(height,1)
                chosed = T.zeros(height).reshape(height,1)
                
                new_tensor = T.cat((detached,rewards,chosed),dim=-1)
                new_tensor.requires_grad_(False)
                
                tensor.data = new_tensor        
                                
    def select_neurons(self):
        tensor_id=0
        
        print(next(self.population[0].parameters()))
        
        # print(next(self.population[0].parameters()))
        
        for tensor in self.base_network.parameters():
            
            if len(tensor.shape)<=1:
                continue
            
            for row_id,row in enumerate(tensor):
                
                with T.no_grad():
            
                    picked_network = choice(self.population)
                    
                    picked_tensor = picked_network.parameters()
                    first_network = self.population[0].parameters()
                    
                    for i,(pop_tensor,first_tensor) in enumerate(zip(picked_tensor,first_network)):
                        if i == tensor_id:
                            choosen_tensor = pop_tensor
                            break
                        
                    # swap tensors between networks
                    
                    swap_tensor = first_tensor[row_id]
                                
                    detached_tensor = T.clone(choosen_tensor[row_id][:-2])
                    
                    first_tensor[row_id] = choosen_tensor[row_id]
                    
                    choosen_tensor[row_id] = swap_tensor
                                        
                    tensor[row_id] = detached_tensor
                    
            tensor_id+=2
            
        print(next(self.population[0].parameters()))
            
        
    # return true if population is ready for crossover
    def shuttle(self):
        
        self.select_neurons()
        
        pop_threshold = self.crossover_counters >= int(self.population_size*0.5)
        
        population_id = -1
        
        for id,pop in enumerate(pop_threshold): 
            if pop > 0.0:
                population_id = id
                self.crossover_counters[id] = 0
                break
                
        # count how many neurons are ready for crossover and which population to cross
        # check if population is ready for crossover
                                
        return population_id
    
    # reset rewards
    def reset(self):
        for net in self.population:
            with T.no_grad():
                for tensor in net.parameters():
                    if len(tensor.shape)<=1:
                        continue
                    for row in tensor:
                        row[-1] = 0.0
                        row[-2] = 0.0
    
        self.crossover_counters = np.zeros(len(self.crossover_counters))
        
    def giveReward(self,reward):
        
        with T.no_grad():
            for tensor in self.population[0].parameters():
                rew = reward / tensor.shape[0]
                if len(tensor.shape)<=1:
                    continue
                for i,row in enumerate(tensor):
                    row[-1]+=rew
                    
                    # check if it is ready for crossover
                    if row[-2]<self.trail_number:
                        row[-2]+=1
                    if row[-2]==self.trail_number:
                        row[-2]+=1
                        self.crossover_counters[i] += 1
                        
    def prepareForSave(self):
        with T.no_grad():
            for net in self.population:
                for tensor in net.parameters():
                    if len(tensor.shape)<=1:
                        continue
                    for row in tensor:
                        if row[-2] >= self.trail_number:
                            row[-2] = self.trail_number-1
            
    
    def givePopulation(self):
        return self.population
    
    def save_models(self):
        
        self.prepareForSave()
        
        for network in self.population:
            for x in network:
                x.save_checkpoint()
        
        for x in self.base_network:
            x.save_checkpoint()

    def load_models(self):
        for network in self.population:
            for x in network:
                x.load_checkpoint()
        
        for x in self.base_network:
            x.load_checkpoint()
            
        # self.reset()
    
    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        outputs = self.base_network(state)
        action = self.policy(outputs)[0]
        return action
