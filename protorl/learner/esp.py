from protorl.learner.base import Learner
import numpy as np
import torch as T
from copy import deepcopy
from random import shuffle

from protorl.utils.initializers import he

# it will perform crossover and mutation 

class ESPLearner(Learner):
    def __init__(self, mutation_probability,members_to_keep,mutation_mean=0.0,mutation_std=0.01,
                 tau=1.0, gamma=0.99, lr=1e-4):
        super().__init__(gamma=gamma, tau=tau)
        self.mutation_probability=mutation_probability
        self.gauss = (mutation_mean,mutation_std)
        self.members_to_keep = members_to_keep
    
    def crossover(self,tensor_a,tensor_b):
                
        size = len(tensor_a)-2
        
        cross_point = int(size/2)
        
        output1 = T.cat((tensor_a[cross_point:],tensor_b[:cross_point]),-1)
        output1 = T.cat((tensor_b[cross_point:],tensor_a[:cross_point]),-1) 
         
        return (output1,output1)
    
    def mutation(self,tensor,muation_probability,gauss=(0.0,0.1)):
        for x in tensor:
            if np.random.random() < muation_probability:
                noise = np.random.normal(gauss[0],gauss[1])
                x += noise
            
            
            
            

    # a list of tuple with (reward,network parameters)
    def update(self, transitions,population_to_crosover=0):
        
        # get all the rows
        population_members = []
        
        with T.no_grad():
            for net in transitions:
                offset = population_to_crosover
                for tensor in net.parameters():
                    if len(tensor.shape)<=1:
                        continue
                    
                    if tensor.shape[1]<population_to_crosover:
                        offset -= population_to_crosover
                        continue
                    
                    population_members.append(tensor[population_to_crosover])
                    break
                        
                    # for row in tensor:
                    #     if row_counter == population_to_crosover:
                    #         print(tensor.shape)
                    #         print("Row: ",len(row))
                    #         population_members.append(row)
                    #         break
                                                        
                    #     row_counter +=1
                    
                    # if row_counter == population_to_crosover:
                    #     break
        
            # sort the transitions
            sorted_transistions = sorted(population_members,key = lambda x: x[-1],reverse=True)
            
            # sorted_transistions = sorted_transistions[:self.members_to_keep]
            
            offsprings = []
            
            for i in range(int(self.members_to_keep/2)):
                
                offs = self.crossover(sorted_transistions[i*2],sorted_transistions[i*2 + 1])
                sorted_transistions[i*2][-1] = 0.0
                sorted_transistions[i*2 + 1][-1] = 0.0
                sorted_transistions[i*2][-2] = 0.0
                sorted_transistions[i*2 + 1][-2] = 0.0
                
                offsprings.extend(offs)
                
            for i,off in enumerate(offsprings):
                sorted_transistions[i+self.members_to_keep][:] = off[:]
                sorted_transistions[i+self.members_to_keep][-1] = 0.0
                sorted_transistions[i+self.members_to_keep][-2] = 0.0
                    
                self.mutation(sorted_transistions[i+self.members_to_keep],self.mutation_probability,self.gauss)
                                
            members_left = len(transitions)-len(offsprings)
            
            if members_left>0:
                # offset = self.members_to_keep+len(offsprings)
                for i in range(members_left):
                    
                    width = len(sorted_transistions[-(i+1)])
                    
                    generated_tensor = T.empty(width)
                    
                    T.nn.init.normal_(generated_tensor,0,np.sqrt(2.0/width))
                    
                    sorted_transistions[-(i+1)][:] = generated_tensor[:]     
            
        
        
        
