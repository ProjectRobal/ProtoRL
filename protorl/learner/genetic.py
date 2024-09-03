from protorl.learner.base import Learner
import numpy as np
import torch as T

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
        
        flatten_tensors_a = T.flatten(tensor_a).numpy()
        flatten_tensors_b = T.flatten(tensor_b).numpy()
        
        size = len(flatten_tensors_a)        
        
        cross_point = int(size/2)
                
        output1 = np.concatenate( (flatten_tensors_a[:cross_point],flatten_tensors_b[cross_point:]) )       
        output2 = np.concatenate( (flatten_tensors_b[:cross_point],flatten_tensors_a[cross_point:]) )     
        
        output_tensor1 = T.tensor(output1).reshape(shape)
        output_tensor2 = T.tensor(output2).reshape(shape)
        
        return (output_tensor1,output_tensor2)
    
    @classmethod
    def params_crossover(cls,param_a,param_b):
        
        offspring_a = []
        offspring_b = []
        
        for i in range(len(param_a)):
            offsprings = GeneticLearner.crossover(param_a[i],param_b[i])
            
            offspring_a.append(offsprings[0])
            offspring_b.append(offsprings[1])
        
        return (offspring_a,offspring_b)

    # a list of tuple with (reward,network parameters)
    def update(self, transitions):
        
        # sort the transitions
        sorted_transistions = sorted(transitions,lambda x: x[0],reverse=True)
        
        sorted_transistions = sorted_transistions[:self.members_to_keep]
        
        offsprings = []
        
        for i in range(int(len(sorted_transistions)/2)):
            
            offs = GeneticLearner.params_crossover(sorted_transistions[i*2],sorted_transistions[i*2 + 1])
            
            offsprings.extend(offs)
            
        transitions.clear()
        
        transitions.extend(sorted_transistions)
        
        transitions.extend([(0.0,x) for x in offsprings])
        
            
            
            
            
        
        
        
