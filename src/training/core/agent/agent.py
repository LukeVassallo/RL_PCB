import os#, sys
#sys.path.append('/home/luke/Desktop/semi_autonomous/py/pcb_component_w_vec_distance_v2')
#sys.path.append('/home/luke/Desktop/semi_autonomous/py/env')
# sys.path.append(os.path.join(os.environ["RL_PCB"], "pcb_component_w_vec_distance_v2"))
# sys.path.append(os.path.join(os.environ["RL_PCB"], "env"))

from pcb_vector_utils import compute_sum_of_euclidean_distances_between_pads
# import os
import numpy as np
import gym
from gym import spaces
# from stable_baselines3 import TD3
# from net import FeaturesExtractor, print_model

from core.agent.observation import get_agent_observation
from core.agent.tracker import tracker
# from core.agent.parameters import parameters


from pcbDraw import draw_board_from_board_and_graph_multi_agent
# from PIL import Image
# import cv2 
import datetime 

class agent(gym.Env):
    def __init__(self, parameters):     # This method is called when an object is created. It's purpose is to initialize the object.        
        self.parameters = parameters    

        obs_space = { "los": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
                        "ol": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
                        "dom": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                        "euc_dist": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                        "position": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                        "ortientation": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            }
        self.observation_space = spaces.Dict(obs_space)
        self.action_space = spaces.Box(low=np.array([0,0,0], dtype=np.float32), high=np.array([1,2*np.pi,1], dtype=np.float32))
        # setup stable baselines 3 model
        #filename, file_extension = os.path.splitext(self.parameters.net)

        #if file_extension == ".td3":
        #    #print(self.parameters.net)
        #    self.model = TD3.load(self.parameters.net)
        #else:
        #    print(f'\"{file_extension.upper()}\" model is not currently supported.')

        self.tracker = tracker()

        self.rng = np.random.default_rng(seed=self.parameters.seed)
        
        # self.max_steps = 200
        self.max_steps = self.parameters.max_steps
        self.steps_done = 0
        
        # self.HPWLe = 0 
        # for net_id in self.parameters.nets:
        #     self.HPWLe += self.parameters.graph.calc_hpwl_of_net(net_id, True)
        
        # self.We = self.parameters.We
        #print(f'{self.parameters.node.get_name()} starts with a HPWL of {self.HPWLe}')
        
        self.HPWLe = self.parameters.opt_hpwl
        self.We = self.parameters.opt_euclidean_distance
        
        self.n = self.parameters.n
        self.m = self.parameters.m
        self.p = self.parameters.p
        
        self.penalty_per_remaining_step = 15
        
    def reset(self):
        self.tracker.reset()
        self.steps_done = 0
        #if self.random_initial_location:

        # print(self.parameters.node.get_name(), self.parameters.node.get_pos())
        #return get_agent_observation(parameters=self.parameters, tracker=self.tracker)
        
        self.W = []
        self.HPWL = []
        self.ol_term5 = []
        self.current_We = self.We
        
        self.Wi = compute_sum_of_euclidean_distances_between_pads(self.parameters.node,self.parameters.neighbors,self.parameters.eoi, ignore_power=self.parameters.ignore_power)
        self.HPWLi = 0
        for net_id in self.parameters.nets:
            self.HPWLi += self.parameters.graph.calc_hpwl_of_net(net_id, True)
        self.current_HPWL = self.HPWLe
        
        self.all_w = []
        self.all_hpwl = []
        self.all_weighted_cost = []
        return
    
    #def _step(self, observation, deterministic=True):
        #action, _states = self.model.predict(observation, deterministic=deterministic)
        #return self.step(action)

    def step(self, model, random:bool=False, deterministic:bool =False, rl_model_type:str = "TD3"):
        self.steps_done += 1
        state = get_agent_observation(parameters=self.parameters)
        _state = list(state["los"]) + list(state["ol"]) + state["dom"] + state["euc_dist"] + state["position"] + state["ortientation"]
        
        if random == True:
            action = self.action_space.sample()
            
            if rl_model_type == "TD3":
                model_action = [0,0,0]#action
                model_action[0] = (action[0] - 0.5) * 2
                model_action[1] = (action[1] - np.pi) / np.pi
                model_action[2] = (action[2] - 0.5) * 2
                
            
        else:
            if rl_model_type == "TD3": # Action scaling done here, outside of policy.
                if deterministic == True:     
                    model_action = model.select_action(np.array(_state)) 
                else:                 
                    model_action = (model.select_action(np.array(_state)) 
                            + np.random.normal(0, self.parameters.max_action * self.parameters.expl_noise, size=3)
                            ).clip(-self.parameters.max_action, self.parameters.max_action)
                # convert action 
                # print(self.steps_done, model_action)
                action = model_action
                action = (action + 1) / 2  # [-1, 1] => [0, 1]
                action[1] *= (2 * np.pi)
                
            else: # SAC - scaling done inside policy
                action = model.select_action(np.array(_state), evaluate=deterministic)            
         
        pos = self.parameters.node.get_pos()
        step_scale = (self.parameters.step_size * action[0])
        x_offset = step_scale*np.cos(-action[1])
        y_offset = step_scale*np.sin(-action[1]) 
        angle = (np.int0(action[2] * 4) % 4) * 90.0
        
        
        
        #print(f'action={action}, x_offset={x_offset}, y_offset={y_offset}, angle={angle}')
        
        self.parameters.node.set_pos(tuple([pos[0] + x_offset, pos[1] + y_offset]))
        self.parameters.node.set_orientation(angle)
        
        #return get_agent_observation(parameters=self.parameters, tracker=self.tracker), 0, False, {}    
        next_state = get_agent_observation(parameters=self.parameters)
        reward, done = self.get_reward(next_state) 
        
        if rl_model_type == "TD3":
            return state, next_state, reward, model_action, done 
        else:
            return state, next_state, reward, action, done 
        #return state, next_state, reward, action, done 

    
    def print(self):
        print(f'====== AGENT controlling node {self.parameters.node.get_id()} ({self.parameters.node.get_name()}) ======')
        print(f'Node position      : {self.parameters.node.get_pos()}')
        print(f'Node orientation   : {self.parameters.node.get_orientation()}')
        print(f'Node size          : {self.parameters.node.get_size()}')
        print()
        print(f'Neighbors')
        for i in range(len(self.parameters.neighbors)):
            print(f'  Neigbor {self.parameters.neighbors[i].get_id()}, {self.parameters.neighbors[i].get_name()}')
            
        print()
        
    def get_reward(self, observation):
        done = False
            #m = self.steps_done/self.max_steps
        self.W.append(compute_sum_of_euclidean_distances_between_pads(self.parameters.node,self.parameters.neighbors,self.parameters.eoi, ignore_power=self.parameters.ignore_power))            

        hpwl = 0
        for net_id in self.parameters.nets:
            hpwl += self.parameters.graph.calc_hpwl_of_net(net_id, True)

        self.HPWL.append(hpwl)
            # ol_mult = np.exp(1.5*m - 0.1) - 1
            # ol_mult = np.clip(ol_mult,0,np.inf) # convert negative value to 0
        
            # if np.sum(self.all_ol[-1]) > 1E-6:
            #     self.ol_term5.append(ol_mult * np.sum(self.all_ol[-1]))
            # else:
            #     self.ol_term5.append(0)
            
        if np.sum(observation["ol"]) > 1E-6:
            # self.ol_term5.append( np.clip((1.156578957894737-np.sum(self.all_ol[-1]))/1.156578957894737, 0.0, np.inf) )
            self.ol_term5.append( np.clip((1-np.sum(observation["ol"])/8), 0.0, np.inf) )
        else:
            self.ol_term5.append(1)
            
        if self.W[-1] < self.We and self.ol_term5[-1] == 1:
            
            if self.parameters.log_file is not None:
                f = open(self.parameters.log_file, 'a')
                f.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]} Agent {self.parameters.node.get_name()} ({self.parameters.node.get_id()}) found a better, legal, wirelength target of {np.round(self.W[-1],6)}, originally {np.round(self.We,6)}.\r\n')
                f.close()
            
            self.We = self.W[-1]
            self.parameters.node.set_opt_euclidean_distance(self.W[-1])

            # print(f'Agent {self.parameters.node.get_name()} found a better, legal, wirelength target of {self.W[-1]}.')

        if self.HPWL[-1] < self.HPWLe:
            stack = draw_board_from_board_and_graph_multi_agent(self.parameters.board, self.parameters.graph, node_id=self.parameters.node.get_id(), padding=4)
            
            stack_sum = np.zeros((stack[0].shape[0],stack[0].shape[1]), dtype=np.int)
            for i in range(len(stack)):
                stack_sum += stack[i]

            if np.max(stack_sum) <= 64:
                #if np.max(stack[1]+stack[2]+stack[3]) <= 64:
                # im = Image.fromarray(np.array(stack[0]+1.8*stack[1]+stack[2]+stack[3]))
                # im.save(f"{self.parameters.node.get_name()}_{self.HPWL[-1]}.jpeg")
    
                #print(f'Agent {self.parameters.node.get_name()} found a better, legal, HPWL target of {self.HPWL[-1]}, originally {self.HPWLe}.')
                if self.parameters.log_file is not None:
                    f = open(self.parameters.log_file, 'a')
                    f.write(f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]} Agent {self.parameters.node.get_name()} ({self.parameters.node.get_id()}) found a better, legal, HPWL target of {np.round(self.HPWL[-1],6)}, originally {np.round(self.HPWLe,6)}.\r\n')
                    f.close()
                # cv2.imshow("del",np.array(stack[0]+stack[1]+stack[2]+stack[3]))
                # cv2.waitKey()
                self.HPWLe = self.HPWL[-1]
                self.parameters.node.set_opt_hpwl(self.HPWL[-1])

        reward = 0

        x = np.clip((self.Wi - self.W[-1]) / (self.Wi - self.current_We), -1, 1)
        y = np.clip((self.HPWLi - self.HPWL[-1]) / (self.HPWLi - self.current_HPWL), -1, 1)
        self.all_w.append(x)
        self.all_hpwl.append(y)
        self.all_weighted_cost.append( (self.n*x + self.m*self.ol_term5[-1] + self.p*y)/(self.n+self.m+self.p) )
        # print(x)
        #reward = self.wl_term5.append(np.tan(x*(np.pi/2.1)))
        reward = np.tan( (self.n*x + self.m*self.ol_term5[-1] + self.p*y)/(self.n+self.m+self.p) * np.pi/2.1  )


            # if new_wl_best5:    
            #     if self.steps_done == 1:
            #         print(self.wl_best5, self.wl_term5[-1], self.Wi, self.W[-1], self.Wi)
            #     else:                 
            #         print(self.wl_best5, self.wl_term5[-1], self.W[-2], self.W[-1], self.Wi)

        if (((observation["position"][0] > 1) or
            (observation["position"][0] < 0) or
            (observation["position"][1] > 1) or
            (observation["position"][1] < 0)) and (np.sum(observation["ol"])/8) == 1):
            # reward -= ((self.max_steps-self.steps_done) * 2*(np.exp(1.5*1 - 0.1) - 1))
            reward -= (self.max_steps-self.steps_done) * self.penalty_per_remaining_step
            done = True  

        if (self.steps_done==self.max_steps):
            done = True
        
        return reward, done        
    
    def init_random(self):
        r_pos = self.rng.uniform(low=0.05, high=0.95, size=(2))
        scaled_r_pos = tuple([r_pos[0]*self.parameters.board_width, r_pos[1]*self.parameters.board_height])
        scaled_orientation = np.float64(self.rng.integers(4)*90)    #0, 90, 180, 270
        self.parameters.node.set_pos(scaled_r_pos)
        self.parameters.node.set_orientation(scaled_orientation)
        
    def get_observation_space_shape(self):
        sz = 0
        for key, value in self.observation_space.items():
            sz += value.shape[0]
            
        return sz
