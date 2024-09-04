from protorl.agents.base import Agent


class ESPAgent(Agent):
    def __init__(self, actor, learner, initializer):
        super().__init__(actor=actor, learner=learner)
        
        self.initializer = initializer
                                    

    def choose_action(self, observation):
        action = self.actor.choose_action(observation)
        return action

    def update_networks(self):
        # do crossover and mutation
        self.learner.update(self.actor.givePopulation(),self.population_to_crossover)
 
    def update(self, transitions):
        
        # reward with a corresponding network
        # self.rewards_and_networks.append((transitions,self.actor.dump_parameters()))
        
        self.actor.giveReward(transitions)        
        
        self.population_to_crossover = self.actor.shuttle()
        
        if self.population_to_crossover != -1:
            print("Crossover has begun!")
            self.update_networks()
            self.actor.reset()
        
