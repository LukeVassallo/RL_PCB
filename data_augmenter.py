import graph.graph as graph
import graph.node as node

import logging, sys
import numpy as np

from graph_utils import kicad_rotate_around_point

class dataAugmenter:
    def __init__(self, board_size=[100,100], max_translation = [1,1], goal = [0,0,0], augment_orientation=False, augment_position=True, rng=None):
        self.board_size = board_size
        self.max_translation = max_translation
        self.goal = goal
        self.augment_orientation = augment_orientation
        self.augment_position = augment_position
        self.rng=rng
    def augment_graph(self, grph, idx=0, brd=None, reset=False):
        if reset==True:
            if brd == None:
                logging.error("Cannot reset graph without board object. Please pass board object to \'augment_graph\' method and try again.")
                return -1

            grph.reset()
            grph.set_component_origin_to_zero(brd)
                
            if (grph.components_to_place() > 1):
                logging.error("Netlist contains more than one unplaced component. This is not allowed, please revise the netlist.")
        
        # Generate offset and change in orientation
        if self.augment_position == True:
            if self.rng == None:
                delta_x = np.random.uniform(low=-self.max_translation[0], high=self.max_translation[0])
                delta_y = np.random.uniform(low=-self.max_translation[1], high=self.max_translation[1])
            else:
                delta_x = self.rng.uniform(low=-self.max_translation[0], high=self.max_translation[0])
                delta_y = self.rng.uniform(low=-self.max_translation[1], high=self.max_translation[1])
        else:
            delta_x = 0
            delta_y = 0
            
        if self.augment_orientation == True:
            if self.rng == None:
                delta_theta = np.random.randint(0,4) * 90 
            else:
                delta_theta = self.rng.integers(0,4) * 90                 
        else:
            delta_theta = 0
                    
        # augment goal
        rotated_pos = kicad_rotate_around_point(self.goal[idx][0]+delta_x, self.goal[idx][1]+delta_y, self.board_size[0]/2, self.board_size[1]/2,  delta_theta)
        orientation = self.goal[idx][2]+delta_theta
        
        if orientation >= 360: 
                orientation -= 360
        augmented_goal = [ rotated_pos[0], rotated_pos[1], orientation ]    
        
        
        nn = grph.get_nodes()
        # augment all components in the netlist apart from the current node to place
        for i in range(len(nn)):
            if nn[i].get_isPlaced() == 0:
                continue

            pos = nn[i].get_pos()
            orientation = nn[i].get_orientation()
            
            rotated_pos = kicad_rotate_around_point(pos[0]+delta_x, pos[1]+delta_y, self.board_size[0]/2, self.board_size[1]/2, delta_theta)        
            nn[i].set_pos((rotated_pos[0],rotated_pos[1]))
            orientation += delta_theta
            if orientation >= 360: 
                orientation -= 360
            nn[i].set_orientation(orientation)
            
        return augmented_goal
    
    def set_translation_limits(self, max_translation):
        self.max_translation = max_translation
        
