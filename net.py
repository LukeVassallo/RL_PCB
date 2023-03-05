#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:30:46 2022

@author: luke
"""
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np

class FeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 23, weight_initialization: str = None, arch: str = ""):
        super(FeaturesExtractor, self).__init__(observation_space, features_dim)
        # print(f'weight_initialization={weight_initialization}')
        input_feature_size = 0
        
        for key, subspace in observation_space.items():
            input_feature_size += subspace.shape[0]
                
        self.nn = torch.nn.Sequential()
        
        self.nn.append(torch.nn.Identity())
        # self.nn.append(torch.nn.Linear(in_features=input_feature_size,out_features=features_dim))
        # self.nn.append(torch.nn.Tanh())
        # self.nn.append(torch.nn.Linear(in_features=32,out_features=32))
        # self.nn.append(torch.nn.Tanh())     

        for name, param in self.nn.named_parameters():
            if name.find("weight") >= 0:
                if weight_initialization == "xavier": 
                    # print("initializing xavier")
                    torch.nn.init.xavier_uniform_(param)       
        
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.nn(self.extract_features(observations))
    
    def print(self):
        #print('  ***************** Neural Network Information *****************')
        print('Neural Network Information')
        print(self)
        print("Number of parameters: ", self.get_parameter_count())
        print("")
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def extract_features(self, observation):
        # concat along the last dimension. The resulting vector should have a size of (num_cpu, num_features)
        obs = observation['los']
        obs = torch.cat((obs,observation['ol']),dim=-1)  
        obs = torch.cat((obs,observation['dom']),dim=-1)
        obs = torch.cat((obs,observation['euc_dist']),dim=-1)  
        obs = torch.cat((obs,observation['position']),dim=-1)  
        obs = torch.cat((obs,observation['ortientation']),dim=-1)

        return obs         
 
class FeaturesExtractor_2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, weight_initialization: str = None, arch: str = ""):
        super(FeaturesExtractor_2, self).__init__(observation_space, features_dim)
        # print(f'weight_initialization={weight_initialization}')
        input_feature_size = 0
        
        for key, subspace in observation_space.items():
            input_feature_size += subspace.shape[0]
                
        self.nn = torch.nn.Sequential()
        
        self.nn.append(torch.nn.Identity())
        # self.nn.append(torch.nn.Linear(in_features=input_feature_size,out_features=features_dim))
        # self.nn.append(torch.nn.Tanh())
        # self.nn.append(torch.nn.Linear(in_features=32,out_features=32))
        # self.nn.append(torch.nn.Tanh())     

        for name, param in self.nn.named_parameters():
            if name.find("weight") >= 0:
                if weight_initialization == "xavier": 
                    # print("initializing xavier")
                    torch.nn.init.xavier_uniform_(param)       
        
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.nn(self.extract_features(observations))
    
    def print(self):
        #print('  ***************** Neural Network Information *****************')
        print('Neural Network Information')
        print(self)
        print("Number of parameters: ", self.get_parameter_count())
        print("")
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def extract_features(self, observation):
        # concat along the last dimension. The resulting vector should have a size of (num_cpu, num_features)
        obs = observation['los']
        obs = torch.cat((obs,observation['ol']),dim=-1)  
        obs = torch.cat((obs,observation['indep_direction_vecs']),dim=-1)
        obs = torch.cat((obs,observation['euc_dist']),dim=-1)  
        obs = torch.cat((obs,observation['position']),dim=-1)  
        obs = torch.cat((obs,observation['ortientation']),dim=-1)

        return obs     

class FeaturesExtractor_3(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, weight_initialization: str = None, arch: str = ""):
        super(FeaturesExtractor_3, self).__init__(observation_space, features_dim)
        # print(f'weight_initialization={weight_initialization}')
        input_feature_size = 0
        
        for key, subspace in observation_space.items():
            input_feature_size += subspace.shape[0]
                
        self.nn = torch.nn.Sequential()
        
        self.nn.append(torch.nn.Identity())
        # self.nn.append(torch.nn.Linear(in_features=input_feature_size,out_features=features_dim))
        # self.nn.append(torch.nn.Tanh())
        # self.nn.append(torch.nn.Linear(in_features=32,out_features=32))
        # self.nn.append(torch.nn.Tanh())     

        for name, param in self.nn.named_parameters():
            if name.find("weight") >= 0:
                if weight_initialization == "xavier": 
                    # print("initializing xavier")
                    torch.nn.init.xavier_uniform_(param)       
        
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.nn(self.extract_features(observations))
    
    def print(self):
        #print('  ***************** Neural Network Information *****************')
        print('Neural Network Information')
        print(self)
        print("Number of parameters: ", self.get_parameter_count())
        print("")
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def extract_features(self, observation):
        # concat along the last dimension. The resulting vector should have a size of (num_cpu, num_features)
        obs = observation['los']
        obs = torch.cat((obs,observation['ol']),dim=-1)  
        obs = torch.cat((obs,observation['dom']),dim=-1)
        obs = torch.cat((obs,observation['cosine_distance']),dim=-1)
        obs = torch.cat((obs,observation['euc_dist']),dim=-1)  
        obs = torch.cat((obs,observation['position']),dim=-1)  
        obs = torch.cat((obs,observation['ortientation']),dim=-1)

        return obs     

class FeaturesExtractor_4(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, weight_initialization: str = None, arch: str = ""):
        super(FeaturesExtractor_4, self).__init__(observation_space, features_dim)
        # print(f'weight_initialization={weight_initialization}')
        input_feature_size = 0
        
        for key, subspace in observation_space.items():
            input_feature_size += subspace.shape[0]
                
        self.nn = torch.nn.Sequential()
        
        self.nn.append(torch.nn.Identity())
        # self.nn.append(torch.nn.Linear(in_features=input_feature_size,out_features=features_dim))
        # self.nn.append(torch.nn.Tanh())
        # self.nn.append(torch.nn.Linear(in_features=32,out_features=32))
        # self.nn.append(torch.nn.Tanh())     

        for name, param in self.nn.named_parameters():
            if name.find("weight") >= 0:
                if weight_initialization == "xavier": 
                    # print("initializing xavier")
                    torch.nn.init.xavier_uniform_(param)       
        
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        return self.nn(self.extract_features(observations))
    
    def print(self):
        #print('  ***************** Neural Network Information *****************')
        print('Neural Network Information')
        print(self)
        print("Number of parameters: ", self.get_parameter_count())
        print("")
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def extract_features(self, observation):
        # concat along the last dimension. The resulting vector should have a size of (num_cpu, num_features)
        obs = observation['los']
        obs = torch.cat((obs,observation['ol']),dim=-1)  
        obs = torch.cat((obs,observation['dom']),dim=-1)
        obs = torch.cat((obs,observation['mean_distance_travelled']),dim=-1)
        obs = torch.cat((obs,observation['euc_dist']),dim=-1)  
        obs = torch.cat((obs,observation['position']),dim=-1)  
        obs = torch.cat((obs,observation['ortientation']),dim=-1)

        return obs     

class HeterogenousFeaturesExtractor_1(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, in_channels: int = 1,  weight_initialization: str = None, arch: str = ""):
        super(HeterogenousFeaturesExtractor_1, self).__init__(observation_space, features_dim)
        # print(f'weight_initialization={weight_initialization}')
        input_feature_size = 0
        
        for key, subspace in observation_space.items():
            input_feature_size += subspace.shape[0]
                
        self.nn = torch.nn.Sequential()
        #self.nn.append(torch.nn.Identity())
        
        self.cnn = torch.nn.Sequential()
        self.concat_layer = torch.nn.Sequential()
        
        # self.nn.append(to
        self.cnn.append(torch.nn.Conv2d(in_channels=in_channels,
            out_channels=32,
            kernel_size=7,
            stride=4,
            padding=0
            )
        )
        self.cnn.append(torch.nn.ReLU())
        self.cnn.append(torch.nn.Conv2d(in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=3,
            padding=0
            )
        )
        self.cnn.append(torch.nn.ReLU())
        self.cnn.append(torch.nn.Conv2d(in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=0
            )
        )
        self.cnn.append(torch.nn.ReLU())
        # self.cnn.append(torch.nn.Conv2d(in_channels=32,
        #     out_channels=16,
        #     kernel_size=3,
        #     stride=2,
        #     padding=0
        #     )
        # )
        # self.cnn.append(torch.nn.ReLU())
        self.cnn.append(torch.nn.Flatten())
        
        with torch.no_grad():
            in_features = self.cnn(torch.as_tensor( self.extract_image_feature(observation_space.sample())[None] ).float()).shape[1]
            #print(torch.as_tensor(np.resize(np.zeros((280,280)),(1,280,280))).shape)
            #in_features = self.cnn(torch.as_tensor(np.resize(np.zeros((280,280)),(1,280,280))).float()).shape[1]
                     
        self.cnn.append(torch.nn.Linear(in_features=in_features,out_features=int(features_dim/2)))
        #self.cnn.append(torch.nn.Tanh())     

        self.nn.append(torch.nn.Linear(in_features=15, out_features=int(features_dim/2)))
        #self.nn.append(torch.nn.Tanh())     

        # self.concat_layer.append(torch.nn.Linear(in_features=features_dim*2, out_features=features_dim))
        self.concat_layer.append(torch.nn.Tanh())     

        # self.nn.append(torch.nn.Tanh())
        # self.nn.append(torch.nn.Linear(in_features=32,out_features=32))
        # self.nn.append(torch.nn.Tanh())     

        for name, param in self.nn.named_parameters():
            if name.find("weight") >= 0:
                if weight_initialization == "xavier": 
                    # print("initializing xavier")
                    torch.nn.init.xavier_uniform_(param)       
        
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        

        a = self.nn(self.extract_handcrafted_features(observations))
        b0 = self.extract_image_feature(observations)
        b = self.cnn(b0)
        tmp = torch.concat((a,b), dim=-1)      
        return self.concat_layer(tmp)
    
    def print(self):
        #print('  ***************** Neural Network Information *****************')
        print('Neural Network Information')
        print(self)
        print("Number of parameters: ", self.get_parameter_count())
        print("")
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def extract_handcrafted_features(self, observation):
        # concat along the last dimension. The resulting vector should have a size of (num_cpu, num_features)
        obs = observation['ol']
        obs = torch.cat((obs,observation['dom']),dim=-1)
        obs = torch.cat((obs,observation['euc_dist']),dim=-1)  
        obs = torch.cat((obs,observation['position']),dim=-1)  
        obs = torch.cat((obs,observation['ortientation']),dim=-1)

        return obs            
    
    def extract_image_feature(self, observation):
        return observation["frame"]

class CustomNetwork(torch.nn.Module):
    """
    A custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    
    def __init__(
            self,
            feature_dim: int, 
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
            weight_initialization: str = None,
            net_arch = None,
            activation_fn = None
    ):
        super(CustomNetwork, self).__init__()
        #feature_dim=23
        print(f'weight_initialization={weight_initialization}')
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers = []  # Layer sizes of the network that only belongs to the value network
        
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # removed shared net - Luke @ 2022/08/25
        #self.shared_net = torch.nn.Sequential(
            ## torch.nn.Linear(feature_dim, feature_dim), torch.nn.ReLU()
            #torch.nn.Linear(feature_dim, 32), torch.nn.Tanh(),
            #torch.nn.Linear(32, 32), torch.nn.Tanh()
        #)
        
        if net_arch is not None: 
            print(net_arch)
            pi = net_arch[-1]["pi"]
            vf = net_arch[-1]["vf"]
            print(f'pi={pi}, vf={vf}')
        else:
            pi = [32,32,128,64,64]
            vf = [64,128,64]
            
        self.latent_dim_pi = pi[-1]
        self.latent_dim_vf = vf[-1]
        
        # removed shared net - Luke @ 2022/08/25
        #for name, param in self.shared_net.named_parameters():
            #if name.find("weight") >= 0:
                #if weight_initialization == "xavier": 
                    ## print("initializing xavier")
                    #torch.nn.init.xavier_uniform_(param)

        # Policy Network
        in_size = feature_dim
        modules = [] 
        for layer_size in pi:
            modules.append(torch.nn.Linear(in_size, layer_size))        
            modules.append(torch.nn.Tanh())
            in_size=layer_size
        self.policy_net = torch.nn.Sequential(*modules)
        
        # Policy network
        #self.policy_net = torch.nn.Sequential(
            
            ##torch.nn.Linear(32, 32), torch.nn.Tanh(),
            #torch.nn.Linear(feature_dim, 32), torch.nn.Tanh(),
            #torch.nn.Linear(32, 32), torch.nn.Tanh(),
            #torch.nn.Linear(32, 128), torch.nn.Tanh(),
            #torch.nn.Linear(128, 64), torch.nn.Tanh(),
            #torch.nn.Linear(64, 64), torch.nn.Tanh(),
        #)
        for name, param in self.policy_net.named_parameters():
            if name.find("weight") >= 0:
                if weight_initialization == "xavier": torch.nn.init.xavier_uniform_(param)

        # Value network
        in_size = feature_dim
        modules = [] 
        for layer_size in vf:
            modules.append(torch.nn.Linear(in_size, layer_size))        
            modules.append(torch.nn.Tanh())
            in_size=layer_size
        self.value_net = torch.nn.Sequential(*modules)
        
        # Value network
        #self.value_net = torch.nn.Sequential(
            ##torch.nn.Linear(32, 64), torch.nn.Tanh(),  
            #torch.nn.Linear(feature_dim, 64), torch.nn.Tanh(),  
            #torch.nn.Linear(64, 128), torch.nn.Tanh(),
            #torch.nn.Linear(128, 64), torch.nn.Tanh(),
        #)
        for name, param in self.value_net.named_parameters():
            if name.find("weight") >= 0:
                        if weight_initialization == "xavier": torch.nn.init.xavier_uniform_(param)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # removed shared net - Luke @ 2022/08/25
        #shared_latent = self.shared_net(features)
        #return self.policy_net(shared_latent), self.value_net(shared_latent) 
        return self.policy_net(features), self.value_net(features) 
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        # removed shared net - Luke @ 2022/08/25
        #shared_latent = self.shared_net(features)
        #return self.policy_net(shared_latent)
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        # removed shared net - Luke @ 2022/08/25
        #shared_latent = self.shared_net(features)
        #return self.value_net(shared_latent)
        return self.value_net(features)
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
        weight_initialization: str = None,
        *args,
        **kwargs,
    ):
        self.weight_initialization = weight_initialization  # placed here for a reason. It seems the the method '_build_mlp_extractor' 
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False



    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, weight_initialization=self.weight_initialization, net_arch=self.net_arch, activation_fn=self.activation_fn)


def print_model(model, file=None):
    if file is not None:
        f = open(file, 'a')
        if f.closed == False:
            f.write('\n\nNeural Network Information\n')
            f.write(str(model.policy))
            f.write("\nModel parameters : " + str(sum(p.numel() for p in model.policy.parameters())))
            f.close()
        else:
            print('\nCouldn\'t write to file because file is closed.')

    else:
        print('\nNeural Network Information')
        print(model.policy)
        print("Model parameters : " + str(sum(p.numel() for p in model.policy.parameters())))

