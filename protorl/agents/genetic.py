from protorl.agents.base import Agent


class GeneticAgent(Agent):
    def __init__(self, actor, learner, population_count,initializer):
        super().__init__(actor=actor, learner=learner)
        
        self.population_count = population_count
        self.initializer = initializer
        
        self.rewards_and_networks=[]
        
        self.net_iterator = 0
        
        self.init_networks()
    
    def init_networks(self):
                
        for i in range(self.population_count):
            
            self.actor.init_network(self.initializer)
            
            self.rewards_and_networks.append((0.0,self.actor.dump_parameters()))
            

        self.net_iterator = 0
                    
        self.actor.load_parameters(self.rewards_and_networks[self.net_iterator][1])
            

    def choose_action(self, observation):
        action = self.actor.choose_action(observation)
        return action

    def update_networks(self):
        # do crossover and mutation
        self.learner.update(self.rewards_and_networks)
 
    def update(self, transitions):
        
        # reward with a corresponding network
        # self.rewards_and_networks.append((transitions,self.actor.dump_parameters()))
        
        self.rewards_and_networks[self.net_iterator][0] = transitions
        
        self.net_iterator+=1
        
        if self.net_iterator >= self.population_count:
            self.update_networks()
            self.net_iterator = 0
        else:                        
            self.actor.load_parameters(self.rewards_and_networks[self.net_iterator][1])
        
