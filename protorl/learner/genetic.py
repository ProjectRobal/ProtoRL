from protorl.learner.base import Learner
import numpy as np
import torch as T
from copy import deepcopy
from random import shuffle

from protorl.utils.initializers import he

import os

# it will perform crossover and mutation 

class GeneticLearner(Learner):
    
    BASE_GENETIC_NAME = "genetic_base"
    HEAD_GENETIC_NAME = "genetic_head"
    
    def __init__(self,network, mutation_probability,members_to_keep,mutation_mean=0.0,mutation_std=0.01,
                 tau=1.0, gamma=0.99, lr=1e-4):
        super().__init__(gamma=gamma, tau=tau)
        
        self.network = network
        
        self.mutation_probability=mutation_probability
        self.gauss = (mutation_mean,mutation_std)
        self.members_to_keep = members_to_keep
    
    def crossover(self,tensor_a,tensor_b):
        
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
    
    def network_crossover(self,param_a,param_b):

        
        offsprings_param_a = []
        offsprings_param_b = []
        
        for a,b in zip(param_a,param_b):
            offsprings = self.crossover(a,b)
             
            offsprings_param_a.append(offsprings[0])
            offsprings_param_b.append(offsprings[1])

        
        return (offsprings_param_a,offsprings_param_b)
    
    def mutation(self,network,muation_probability,gauss=(0.0,0.1)):
        
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
            
            
    def load_network_with_id(self,id):
        
        checkpoint_dir = self.network[0].checkpoint_dir
        
        base_name = self.BASE_GENETIC_NAME + "_{}".format(id)
        head_name = self.HEAD_GENETIC_NAME + "_{}".format(id)
        
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
        
    def save_network_with_id(self,id):
        
        checkpoint_dir = self.network[0].checkpoint_dir
        
        for net in self.network:
            net.save_checkpoint()
        
        base_name = self.BASE_GENETIC_NAME + "_{}".format(id)
        head_name = self.HEAD_GENETIC_NAME + "_{}".format(id)
        
        base_path = os.path.join(checkpoint_dir,base_name)
        head_path = os.path.join(checkpoint_dir,head_name)
        
        parent_base_path = os.path.join(checkpoint_dir,self.BASE_GENETIC_NAME)
        parent_head_path = os.path.join(checkpoint_dir,self.HEAD_GENETIC_NAME)
        
        os.rename(parent_base_path,base_path)
        os.rename(parent_head_path,head_path)

    # a list of tuple with (reward,network parameters)
    def update(self, transitions):
        
        # sort the transitions
        sorted_transistions = sorted(transitions,key = lambda x: x[0],reverse=True)
        
        # sorted_transistions = sorted_transistions[:self.members_to_keep]
        
        offsprings = []
        
        layer_count = len(self.network)
        
        cross_point = int(self.members_to_keep/2)
        
        for i in range(cross_point):
            
            # load network A
            # load it's i parameter
            # load network B
            # load it's i parameter
            # perform a crossover
            # load network child A
            # set appropiate parameter
            # load network child B
            # set appropiate parameter
            # save network child A
            # save network child B
            
            parent_a = sorted_transistions[i*2][1]
            parent_b = sorted_transistions[i*2 + 1][1]
            
            for layer_id in range(layer_count):
            
                self.load_network_with_id(parent_a)
                
                paramA = [] 
                
                for param in self.network[layer_id].parameters():
                    paramA.append(param)
                
                self.load_network_with_id(parent_b)
                
                paramB = []
                
                for param in self.network[layer_id].parameters():
                    paramB.append(param)
                
                childA,childB = self.network_crossover(paramA,paramB)
                
                #save networks
                                
                for p,param in enumerate(self.network[layer_id].parameters()):
                    param.data.copy_(childA[p])
                    
                self.save_network_with_id(parent_a + cross_point)
                
                for p,param in enumerate(self.network[layer_id].parameters()):
                    param.data.copy_(childB[p])
                                
                self.save_network_with_id(parent_b + cross_point)                
                
            
            # offs = self.network_crossover(sorted_transistions[i*2][1].parameters(),sorted_transistions[i*2 + 1][1].parameters())
            
            # offsprings.extend(offs)
            
        for i in range(cross_point,cross_point*2,1):
            self.load_network_with_id(i)
                
            self.mutation(self.network,self.mutation_probability)
                
            self.save_network_with_id(i)
                
            
        # for i,off in enumerate(offsprings):
        #     network = sorted_transistions[i+self.members_to_keep][1]
            
        #     params = network.parameters()
            
        #     for p,param in enumerate(params):
        #         param.data.copy_(off[p])
                
        #     self.mutation(network,self.mutation_probability,self.gauss)
            
        members_left = len(transitions)-cross_point*2
        
        if members_left>0:
            # offset = self.members_to_keep+len(offsprings)
            # for i in range(members_left):
            #     sorted_transistions[-(i+1)][1].apply(he)
            
            for i in range(members_left):
                self.load_network_with_id(cross_point*2 + i)
                self.network.apply(he)
                self.save_network_with_id(cross_point*2 + i)
        
        
        shuffle(transitions)
            
            
        
        
        
