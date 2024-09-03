from protorl.learner.base import Learner
import numpy as np
import torch as T
from copy import deepcopy
from random import shuffle

from protorl.utils.initializers import he

# it will perform crossover and mutation 

class GeneticLearner(Learner):
    def __init__(self, mutation_probability,members_to_keep,mutation_mean=0.0,mutation_std=0.01,
                 tau=1.0, gamma=0.99, lr=1e-4):
        super().__init__(gamma=gamma, tau=tau)
        self.mutation_probability=mutation_probability
        self.gauss = (mutation_mean,mutation_std)
        self.members_to_keep = members_to_keep
    
    @classmethod
    def crossover(cls,tensor_a,tensor_b):
        
        shape = tensor_a.shape
        
        flatten_tensors_a = T.flatten(tensor_a).detach().numpy()
        flatten_tensors_b = T.flatten(tensor_b).detach().numpy()
        
        size = len(flatten_tensors_a)        
        
        cross_point = int(size/2)
                
        output1 = np.concatenate( (flatten_tensors_a[:cross_point],flatten_tensors_b[cross_point:]) )       
        output2 = np.concatenate( (flatten_tensors_b[:cross_point],flatten_tensors_a[cross_point:]) )     
        
        output_tensor1 = T.tensor(output1).reshape(shape)
        output_tensor2 = T.tensor(output2).reshape(shape)
        
        return (output_tensor1,output_tensor2)
    
    @classmethod
    def network_crossover(cls,param_a,param_b):

        
        offsprings_param_a = []
        offsprings_param_b = []
        
        for a,b in zip(param_a,param_b):
            offsprings = GeneticLearner.crossover(a,b)
            
            offsprings_param_a.append(offsprings[0])
            offsprings_param_b.append(offsprings[1])

        
        return (offsprings_param_a,offsprings_param_b)
    
    @classmethod
    def mutation(cls,network,muation_probability,gauss=(0.0,0.1)):
        
        params = network.parameters()
        
        for param in params:
            
            shape = param.shape
            
            w = T.flatten(param).detach().numpy()
            
            for x in w:
                if np.random.random() < muation_probability:
                    noise = np.random.normal(gauss[0],gauss[1])
                    x += noise
            
            mutated = T.tensor(w).reshape(shape)
            
            param.data.copy_(mutated)
            
            
            

    # a list of tuple with (reward,network parameters)
    def update(self, transitions):
        
        # sort the transitions
        sorted_transistions = sorted(transitions,key = lambda x: x[0],reverse=True)
        
        # sorted_transistions = sorted_transistions[:self.members_to_keep]
        
        offsprings = []
        
        for i in range(int(self.members_to_keep/2)):
            
            offs = GeneticLearner.network_crossover(sorted_transistions[i*2][1].parameters(),sorted_transistions[i*2 + 1][1].parameters())
            
            offsprings.extend(offs)
            
        for i,off in enumerate(offsprings):
            network = sorted_transistions[i+self.members_to_keep][1]
            
            params = network.parameters()
            
            for p,param in enumerate(params):
                param.data.copy_(off[p])
                
            GeneticLearner.mutation(network,self.mutation_probability,self.gauss)
            
        members_left = len(transitions)-len(offsprings)
        
        if members_left>0:
            # offset = self.members_to_keep+len(offsprings)
            for i in range(members_left):
                sorted_transistions[-(i+1)][1].apply(he)
        
        
        shuffle(transitions)
            
            
            
            
        
        
        
