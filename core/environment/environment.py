import os, sys
#sys.path.append('/home/luke/Desktop/semi_autonomous/py')
sys.path.append(os.path.join(os.environ["RL_PCB"], "py"))
from data_augmenter import dataAugmenter
import pcb.pcb as pcb
import graph.graph as graph     # Necessary for graph related methods

from core.agent.agent import agent as agent
from core.agent.parameters import parameters as agent_parameters
from core.environment.tracker import tracker


from pcbDraw import draw_board_from_board_and_graph_with_debug, draw_ratsnest_with_board

import numpy as np
import os
import random as random_package

class environment:
    def __init__(self, parameters):     # This method is called when an object is created. It's purpose is to initialize the object.
        self.parameters = parameters

        self.pv = pcb.vptr_pcbs()
        pcb.read_pcb_file(self.parameters.pcb_file, self.pv)      # Read pcb file
        self.rng = np.random.default_rng(seed=self.parameters.seed)
        self.initialize_environment_state_from_pcb(init=True, idx=self.parameters.idx)   # sets p,g and b variables; idx=None => random!!
        self.tracker = tracker()
                
        if self.parameters.use_dataAugmenter == True:
            # The following configures the maximum translation. The following constraints / requirements apply:
            #   1. There must be only one LOCKED component in the netlist.
            #   2. The LOCKED component must be centered in the middle of the board  
            nn = self.g.get_nodes()
            c_sz = 0
            b_sz = np.minimum(self.b.get_width(), self.b.get_height())
            for i in range(len(nn)):
                if nn[i].get_isPlaced() == 1:
                    c_sz = np.maximum(nn[i].get_size()[0], nn[i].get_size()[1])
                    break
                    
            sz = (b_sz - c_sz) / 2.0
                    
            translation_limits = [0.66*sz, 0.66*sz]
 
            self.dA = dataAugmenter(board_size=[self.b.get_width(), self.b.get_height()],
                       max_translation=translation_limits,
                       goal=[[-1,-1, 0]],
                       augment_position=self.parameters.augment_position,
                       augment_orientation=self.parameters.augment_orientation,
                       rng = self.rng)
                
        #self.We.append(compute_sum_of_euclidean_distances_between_pads(self.node, self.neighbors, self.eoi))
        self.padding=4

    def reset(self):        
        self.g.update_original_nodes_with_current_optimals()
        self.initialize_environment_state_from_pcb(init=True, idx=self.parameters.idx)   # sets p,g and b variables; idx=None => random!!; set to True for multiple PCBs
        #self.initialize_environment_state_from_pcb()
        
        if self.parameters.use_dataAugmenter == True:
            # The following configures the maximum translation. The following constraints / requirements apply:
            #   1. There must be only one LOCKED component in the netlist.
            #   2. The LOCKED component must be centered in the middle of the board  
            nn = self.g.get_nodes()
            c_sz = 0
            b_sz = np.minimum(self.b.get_width(), self.b.get_height())
            for i in range(len(nn)):
                if nn[i].get_isPlaced() == 1:
                    c_sz = np.maximum(nn[i].get_size()[0], nn[i].get_size()[1])
                    break
                    
            sz = (b_sz - c_sz) / 2.0
            
            self.dA.board_size=[self.b.get_width(), self.b.get_height()]
            self.dA.set_translation_limits([0.66*sz, 0.66*sz])
            self.optimal_location = self.dA.augment_graph(grph=self.g, idx=0)   # self.optimals and index are not used.

        for i in range(len(self.agents)):
            self.agents[i].init_random()    
        
        for i in range(len(self.agents)):   # Computes the wirelength and HPWL
            self.agents[i].reset()     

        #comp_grids = draw_board_from_nodes_multi_agent(self.nodes, self.b.get_width(), self.b.get_height(), padding=4)
        if self.parameters.debug:
            comp_grids = draw_board_from_board_and_graph_with_debug(self.b, self.g, padding=self.padding)
            for i in range(len(self.agents)):
                if i == 0:
                    ratsnest = draw_ratsnest_with_board(self.agents[i].parameters.node, self.agents[i].parameters.neighbors, self.agents[i].parameters.eoi, self.b, line_thickness=1, padding=self.padding, ignore_power=True)
                else:
                    ratsnest = np.maximum(ratsnest, draw_ratsnest_with_board(self.agents[i].parameters.node, self.agents[i].parameters.neighbors, self.agents[i].parameters.eoi, self.b, line_thickness=1, padding=self.padding, ignore_power=True))

            self.tracker.add(comp_grids=comp_grids,ratsnest=ratsnest)

        return

    # def _step(self, observation, deterministic=True):
    #     # print(observation)
    #     print()
    #     obsNstuff = []
    #     for i in range(len(self.all_agents)):
    #         # print('reset', a.parameters.node.get_name(), hex(id(a.parameters.node)))
    #         # print(observation[i])
    #         obsNstuff.append(self.all_agents[i]._step(observation=observation[i], deterministic=deterministic))

    #     comp_grids = draw_board_from_board_and_graph(self.b, self.g)
    #     self.tracker.add_comp_grids(comp_grids=comp_grids)

    #     return obsNstuff

    def step(self, model, random=False, deterministic:bool = False, rl_model_type:str ="TD3"):
        observation_vec = []
        step_metrics = []
        
        idxs = [] 
        for i in range(len(self.agents)):
            idxs.append(i)
            
        if self.parameters.shuffle_idxs == True:
            random_package.shuffle(idxs)
        
        # for i in range(len(self.agents)):
        for i in idxs:
            state, next_state, reward, action, done = self.agents[i].step(model=model, random=random, deterministic=deterministic, rl_model_type=rl_model_type)
            # convert state_vector
            _state = list(state["los"]) + list(state["ol"]) + state["dom"] + state["euc_dist"] + state["position"] + state["ortientation"]
            _next_state = list(next_state["los"]) + list(next_state["ol"]) + next_state["dom"] + next_state["euc_dist"] + next_state["position"] + next_state["ortientation"]
            _next_state_info = next_state["info"]
            observation_vec.append([_state, _next_state, reward, action, done, _next_state_info])
            
            step_metrics.append({"id": self.agents[i].parameters.node.get_id(),
             "name": self.agents[i].parameters.node.get_name(),
             "reward": reward,
             "W": self.agents[i].all_w[-1],
             "We": self.agents[i].We,
             "HPWL": self.agents[i].all_hpwl[-1],
             "HPWLe": self.agents[i].HPWLe,
             "ol": 1-self.agents[i].ol_term5[-1],
             "weighted_cost": self.agents[i].all_weighted_cost[-1],
             "raw_W": self.agents[i].W[-1],
             "raw_HPWL": self.agents[i].HPWL[-1],
             "Wi": self.agents[i].Wi,
             "HPWLi": self.agents[i].HPWLi,
             })
            
            if done == True:
                break
            
        if self.parameters.debug == True:
            comp_grids = draw_board_from_board_and_graph_with_debug(self.b, self.g, padding=self.padding)  
            for i in range(len(self.agents)):
                if i == 0:
                    ratsnest = draw_ratsnest_with_board(self.agents[i].parameters.node,
                                                        self.agents[i].parameters.neighbors, 
                                                        self.agents[i].parameters.eoi, 
                                                        self.b, line_thickness=1, 
                                                        padding=self.padding, ignore_power=True)
                else:
                    ratsnest = np.maximum(ratsnest, draw_ratsnest_with_board(self.agents[i].parameters.node, 
                                                                             self.agents[i].parameters.neighbors, 
                                                                             self.agents[i].parameters.eoi, 
                                                                             self.b, 
                                                                             line_thickness=1, 
                                                                             padding=self.padding,
                                                                             ignore_power=True))

            self.tracker.add(comp_grids=comp_grids,ratsnest=ratsnest)

        self.tracker.add_metrics(step_metrics)
        return observation_vec
        # return observation, reward, done, {}
        # for i in range(len(self.all_agents)):
        #     state, next_state, reward, action, done = self.all_agents[i].step()
        #     if done: break
        


    def initialize_environment_state_from_pcb(self, init = False, idx=None):
        #self.p = self.pv[0]                             # Assuming one pcb.
        if idx==None:
            self.idx = int(self.rng.integers(len(self.pv)))
        else:
            self.idx = idx
        # print(f"Loading pcb with idx={self.idx}")
        self.p = self.pv[self.idx]
        
        if init: self.agents = []
        self.g = self.p.get_graph()
        self.g.reset()
        self.b = self.p.get_board()
        self.g.set_component_origin_to_zero(self.b)   # >>> VERY VERY IMPORTANT <<<
       
        nn = self.g.get_nodes()
        for i in range(len(nn)):
            if nn[i].get_isPlaced() == 0:
                node_id = nn[i].get_id()
                #node = self.g.get_node_by_id(node_id)
                nets = []
    
                neighbor_ids = self.g.get_neighbor_node_ids(node_id)
                neighbors = []
                for n_id in neighbor_ids:
                    neighbors.append(self.g.get_node_by_id(n_id))
    
                ee = self.g.get_edges()
                eoi = []
                for e in ee:
                    if e.get_instance_id(0) == node_id or e.get_instance_id(1) == node_id:
                        eoi.append(e)
                        nets.append(e.get_net_id())  
                        
                if init:
                    #print(nn[i].get_id(),nn[i].get_name())
                    agent_params = agent_parameters({"board": self.b,
                                                     "graph": self.g,
                                                     "board_width": self.b.get_width(),
                                                     "board_height": self.b.get_height(),
                                                     "node": nn[i],
                                                     "neighbors": neighbors,
                                                     "eoi": eoi,
                                                     "nets": set(nets),
                                                     "net": self.parameters.net,
                                                     "seed": self.rng.integers(0,65535),
                                                     "step_size": 1.0,
                                                     "max_steps": self.parameters.max_steps,
                                                     "expl_noise": self.parameters.agent_expl_noise,
                                                     "max_action": self.parameters.agent_max_action,
                                                     #"We": np.sqrt(np.square(self.b.get_width())+np.square(self.b.get_height())),
                                                     "opt_euclidean_distance": nn[i].get_opt_euclidean_distance(),
                                                     "opt_hpwl": nn[i].get_opt_hpwl(),
                                                     "n": self.parameters.n,
                                                     "m": self.parameters.m,
                                                     "p": self.parameters.p,
                                                     "ignore_power": self.parameters.ignore_power,
                                                     "log_file": None if self.parameters.log_dir == None else os.path.join(self.parameters.log_dir, self.p.get_kicad_pcb2().replace(".kicad_pcb", ".log")),
                                                    })
                    
                    self.agents.append(agent(agent_params))    
                    
                else: # if init:
                    self.agents[i].parameters.node = nn[i]
                    self.agents[i].parameters.neighbors = neighbors
                    self.agents[i].parameters.eoi = eoi
                

    def get_target_params(self):
        target_params = []
        for agent in self.agents:
            target_params.append({"id": agent.parameters.node.get_id(),
             "We": agent.We,
             "HPWLe": agent.HPWLe })
            
        return target_params
            
    def get_all_target_params(self):
        original_idx = self.idx
        all_params = []
        for i in range(len(self.pv)):        
            self.initialize_environment_state_from_pcb(init = True, idx=i)
            all_params.append({"kicad_pcb":self.p.get_kicad_pcb2(), "expert_targets": self.get_target_params()})
            
        self.initialize_environment_state_from_pcb(init = True, idx=original_idx)
        return all_params

            
    def info(self):
        """
        Iterates through all agents and prints their information

        Returns
        -------
        None.

        """
        
        for agent in self.agents:
            agent.print()
        
    def library_info(self):
        graph.build_info()
        pcb.build_info()
        
    def library_info_as_string(self):
        s = "<strong>====== Library information ======</strong><br>"
        s += pcb.build_info_as_string().replace('\n', '<br>')#[:-4]      # strips last 4 characters to prevent printing double '\n'
        s += "pcb library dependency #1<br>"
        s += pcb.dependency_info_as_string().replace('\n', '<br>')[4:-4] # strips first and last 4 characters to prevent printing double '\n'
        s += graph.build_info_as_string().replace('\n', '<br>')
        return s
        
    def write_pcb_file(self, path=None, filename=None):
        
        if path != None and filename != None:
            save_loc = os.path.join(path, filename)
        else:
            save_loc = "./pcb_file.pcb"
            
        for i in range(len(self.pv)):
            g = self.pv[i].get_graph()
            g.reset()
        pcb.write_pcb_file(save_loc, self.pv, False)
        
    def write_current_pcb_file(self, path=None, filename=None):
        
        if path != None and filename != None:
            save_loc = os.path.join(path, filename)
        else:
            save_loc = "./pcb_file.pcb"
             
        pv = pcb.vptr_pcbs()
        pv.append(self.pv[self.idx])
        g = pv[0].get_graph()
        g.update_hpwl(do_not_ignore_unplaced=True)
        g.reset_component_origin(self.b)   # >>> VERY VERY IMPORTANT <<<
        pcb.write_pcb_file(save_loc, pv, False)      # append = False    
        g.set_component_origin_to_zero(self.b)   # >>> VERY VERY IMPORTANT <<<

    def calc_hpwl(self):
        return self.g.calc_hpwl(True)
    
    def get_parameters(self):
        return self.parameters
    
    def get_current_pcb_name(self):
        # drop '.kicad_pcb'
        return self.pv[self.idx].get_kicad_pcb2().split('.')[0]
