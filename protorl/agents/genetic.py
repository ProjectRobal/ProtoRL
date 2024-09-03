from protorl.agents.base import Agent


class GeneticAgent(Agent):
    def __init__(self, actor, learner, initializer):
        super().__init__(actor=actor, learner=learner)
        
        self.initializer = initializer
                                    

    def choose_action(self, observation):
        action = self.actor.choose_action(observation)
        return action

    def update_networks(self):
        # do crossover and mutation
        self.learner.update(self.actor.givePopulation())
        self.population_count = len(self.actor.givePopulation())
 
    def update(self, transitions):
        
        # reward with a corresponding network
        # self.rewards_and_networks.append((transitions,self.actor.dump_parameters()))
        
        self.actor.giveReward(transitions)        
        
        if self.actor.shuttle():
            print("Crossover has begun!")
            self.update_networks()
            self.actor.reset()
        
