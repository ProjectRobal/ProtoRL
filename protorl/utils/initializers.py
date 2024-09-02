import torch as T

def uniform(network,a,b):
    
    def uniform_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.uniform_(m.weight,a,b)
            T.nn.init.uniform_(m.bias,a,b)
            
    network.apply(uniform_init)
    
def normal(network,mean,std):
    
    def normal_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.normal_(m.weight,mean,std)
            T.nn.init.normal_(m.bias,mean,std)
            
    network.apply(normal_init)
    
def constant(network,val):
    
    def constant_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.constant_(m.weight,val)
            T.nn.init.constant_(m.bias,val)
            
    network.apply(constant_init)
    
def ones(network):
    
    def ones_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.ones_(m.weight)
            T.nn.init.ones_(m.bias)
            
    network.apply(ones_init)
    
def zeros(network):
    
    def zeros_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.zeros_(m.weight)
            T.nn.init.zeros_(m.bias)
            
    network.apply(zeros_init)
    
def xavier(network,gain=1.0):
    
    def xavier_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.xavier_uniform_(m.weight,gain)
            T.nn.init.xavier_uniform_(m.bias,gain)
            
    network.apply(xavier_init)
    
def he(network,gain=1.0):
    
    def he_init(m):
        if isinstance(m, T.nn.Linear):
            T.nn.init.xavier_normal_(m.weight,gain)
            T.nn.init.xavier_normal_(m.bias,gain)
            
    network.apply(he_init)