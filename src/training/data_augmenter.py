"""
This module provides a data augmentation functionality for graphs.

The `dataAugmenter` class in this module allows for augmenting a graph by
applying translations and rotations to its components. It provides methods to
perform graph augmentation and set translation limits.

Module Dependencies:
    - logging
    - numpy
    - graph_utils.kicad_rotate_around_point

Classes:
    - dataAugmenter: A class for augmenting graphs with translations and
    rotations.

Example Usage:
    augmenter = dataAugmenter(board_size=[200, 200], max_translation=[2, 2])
    augmented_goal = augmenter.augment_graph(graph, idx=0, brd=board,
    reset=True)

"""
import logging
import numpy as np

import graph.graph as graph
import graph.board as board
import graph.node as node
import graph.edge as edge
from graph_utils import kicad_rotate_around_point


class dataAugmenter:
    """A class that performs data augmentation on a graph.

    This class provides methods to augment a graph by applying translations
    and rotations to its components.

    Args:
        board_size (list): The size of the board [width, height].
                           Default is [100, 100].
        max_translation (list): The maximum translation allowed in each axis
                                [x, y]. Default is [1, 1].
        goal (list): The goal position for augmentation [x, y, orientation].
                     Default is [0, 0, 0].
        augment_orientation (bool): Whether to augment the orientation of
                                    components. Default is False.
        augment_position (bool): Whether to augment the position of components.
                                 Default is True.
        rng (numpy.random.Generator): The random number generator.
                                      Default is None.
    """
    def __init__(self,
                 board_size=[100,100],
                 max_translation = [1,1],
                 goal = [0,0,0],
                 augment_orientation=False,
                 augment_position=True,
                 rng=None):
        self.board_size = board_size
        self.max_translation = max_translation
        self.goal = goal
        self.augment_orientation = augment_orientation
        self.augment_position = augment_position
        self.rng=rng
    def augment_graph(self, grph, idx=0, brd=None, reset=False):
        if reset is True:
            if brd is None:
                logging.error("Cannot reset graph without board object. "
                              "Please pass board object to 'augment_graph' "
                              "method and try again.")
                return -1

            grph.reset()
            grph.set_component_origin_to_zero(brd)

            if grph.components_to_place() > 1:
                logging.error("Netlist contains more than one unplaced "
                              "component. This is not allowed, please revise "
                              "the netlist.")

        # Generate offset and change in orientation
        if self.augment_position is True:
            if self.rng is None:
                delta_x = np.random.uniform(low=-self.max_translation[0],
                                            high=self.max_translation[0])
                delta_y = np.random.uniform(low=-self.max_translation[1],
                                            high=self.max_translation[1])
            else:
                delta_x = self.rng.uniform(low=-self.max_translation[0],
                                           high=self.max_translation[0])
                delta_y = self.rng.uniform(low=-self.max_translation[1],
                                           high=self.max_translation[1])
        else:
            delta_x = 0
            delta_y = 0

        if self.augment_orientation is True:
            if self.rng is None:
                delta_theta = np.random.randint(0,4) * 90
            else:
                delta_theta = self.rng.integers(0,4) * 90
        else:
            delta_theta = 0

        # augment goal
        rotated_pos = kicad_rotate_around_point(self.goal[idx][0]+delta_x,
                                                self.goal[idx][1]+delta_y,
                                                self.board_size[0]/2,
                                                self.board_size[1]/2,
                                                delta_theta)
        orientation = self.goal[idx][2]+delta_theta

        if orientation >= 360:
            orientation -= 360
        augmented_goal = [ rotated_pos[0], rotated_pos[1], orientation ]

        nn = grph.get_nodes()
        # augment all components in the netlist apart from the current node
        # to place
        for i in range(len(nn)):
            if nn[i].get_isPlaced() == 0:
                continue

            pos = nn[i].get_pos()
            orientation = nn[i].get_orientation()

            rotated_pos = kicad_rotate_around_point(pos[0]+delta_x,
                                                    pos[1]+delta_y,
                                                    self.board_size[0]/2,
                                                    self.board_size[1]/2,
                                                    delta_theta)
            nn[i].set_pos((rotated_pos[0],rotated_pos[1]))
            orientation += delta_theta
            if orientation >= 360:
                orientation -= 360
            nn[i].set_orientation(orientation)

        return augmented_goal

    def set_translation_limits(self, max_translation):
        self.max_translation = max_translation
