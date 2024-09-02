from protorl.agents.base import Agent


class GeneticAgent(Agent):
    def __init__(self, actor, learner, population_count,initializer):
        super().__init__(actor=actor, learner=learner)
        
        self.population_count = population_count
        self.initializer = initializer
        
        self.rewards=[]
        self.past_networks=[]
    
    def init_network(self):
        
        network = []
        
        self.actor.init_network(self.initializer)

    def choose_action(self, observation):
        action = self.actor.choose_action(observation)
        return action

    def update_networks(self):
        # do crossover and mutation
        pass 

    def update(self, transitions):
        
        self.rewards.append(transitions)
        
        if len(self.rewards) > self.population_count:
            self.update_networks()
        else:
            # generate new weights for actor
            self.actor.networks
        
