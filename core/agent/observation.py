#import sys
#sys.path.append('/home/luke/Desktop/semi_autonomous/py')
import os, sys
sys.path.append(os.environ["RL_PCB"])

from pcbDraw import draw_los, draw_board_from_nodes_and_edges, draw_board_from_nodes_and_edges_multi_agent, draw_board_from_board_and_graph_multi_agent, draw_board_from_graph_multi_agent, draw_ratsnest, get_los_and_ol, get_los_and_ol_multi_agent
from pcb_vector_utils import compute_pad_referenced_distance_vectors_v2, compute_vector_to_group_midpoint
from pcb_vector_utils import wrap_angle
import numpy as np

def line_of_sight_and_overlap_v0(parameters, comp_grids):
    los_grids = []
    los = []

    ol_grids = []
    ol = []

    current_node_pos = parameters.node.get_pos()
    current_node_size = parameters.node.get_size()
    current_orientation = parameters.node.get_orientation()

    los_segments, segment_pixels = draw_los(current_node_pos[0],
                                    current_node_pos[1],
                                    np.max(current_node_size)*1.5,
                                        angle_offset=current_orientation,
                                        bx=parameters.board_width,
                                        by=parameters.board_height,
                                        padding=parameters.padding)

    for i in range(8):      # Calculate line-of-sight
        los_grids.append(comp_grids[0] * (los_segments[i]/16))
        los.append(((np.sum(los_grids[-1])/64) + 1E-3) / segment_pixels[i])

    for i in range(8):      # Calculate overlap between current component and placed components
        ol_grids.append((comp_grids[0]/64) * (comp_grids[1]/64) * (los_segments[i]/16))
        ol.append((np.sum(ol_grids[-1]) + 1E-6) / np.sum(comp_grids[1]/64))        

    return los_grids, los, ol_grids, ol

   
def get_agent_observation(parameters, tracker=None):

    node_id = parameters.node.get_id()
    # draw comp_grid from nodes
    #comp_grids = draw_board_from_nodes_and_edges(n=parameters.node, nn=parameters.neighbors, e=parameters.eoi, bx=parameters.board_width, by=parameters.board_height, padding=parameters.padding)  
    #comp_grids = draw_board_from_nodes_and_edges_multi_agent(n=parameters.node, nn=parameters.neighbors, e=parameters.eoi, bx=parameters.board_width, by=parameters.board_height, padding=parameters.padding)  
    comp_grids = draw_board_from_graph_multi_agent(g=parameters.graph, node_id=node_id, bx=parameters.board_width, by=parameters.board_height, padding=parameters.padding)

    #los_grids, los, ol_grids, ol = line_of_sight_and_overlap_v0(parameters=parameters, comp_grids=comp_grids)
    los, ol, los_grids, ol_grids = get_los_and_ol_multi_agent(node=parameters.node, board=parameters.board, radius=np.max(parameters.node.get_size())*1.5, grid_comps=comp_grids, padding=parameters.padding)
        
    # all_comp_grids.append(comp_grids)
    # self.all_los_grids.append(los_grids)
    # self.all_los.append(los)
    # self.all_ol_grids.append(ol_grids)
    # self.all_ol.append(ol)
    
    # compute ol_ratio
    ol_ratios = []
    total = np.sum(ol_grids)/64
    for grid in ol_grids:
        ol_ratios.append((np.sum(grid) / 64) / total)

    dom, resultant, all_vecs = compute_pad_referenced_distance_vectors_v2(
        parameters.node, 
        parameters.neighbors, 
        parameters.eoi,
        ignore_power=parameters.ignore_power_nets
        )

    # group
    mp, eucledian_dist, angle = compute_vector_to_group_midpoint(
        parameters.node, 
        parameters.neighbors
    )            

    if tracker is not None:
        tracker.add_observation(comp_grids=comp_grids)
        tracker.add_ratsnest(draw_ratsnest(parameters.node, parameters.neighbors, parameters.eoi, parameters.board_width, parameters.board_height, padding=parameters.padding, ignore_power=parameters.ignore_power_nets))

    info = { "ol_ratios": ol_ratios, }

    return { "los":  los[-8:],
                    "ol": ol[-8:],
                    "dom": [dom[0], dom[1]],
                    "euc_dist": [ eucledian_dist, angle ],
                    "position": [parameters.node.get_pos()[0] / parameters.board_width, parameters.node.get_pos()[1] / parameters.board_height],                    
                    "ortientation": [wrap_angle(parameters.node.get_orientation())],
                    "info": info
        }        

# The following is the original function before making any change to overalp. Delete when previous in working.
# def get_agent_observation(parameters, tracker=None):

#     # draw comp_grid from nodes
#     #comp_grids = draw_board_from_nodes_and_edges(n=parameters.node, nn=parameters.neighbors, e=parameters.eoi, bx=parameters.board_width, by=parameters.board_height, padding=parameters.padding)  
#     comp_grids = draw_board_from_nodes_and_edges_multi_agent(n=parameters.node, nn=parameters.neighbors, e=parameters.eoi, bx=parameters.board_width, by=parameters.board_height, padding=parameters.padding)  

#     #los_grids, los, ol_grids, ol = line_of_sight_and_overlap_v0(parameters=parameters, comp_grids=comp_grids)
#     los, ol, los_grids, ol_grids = get_los_and_ol_multi_agent(node=parameters.node, board=parameters.board, radius=np.max(parameters.node.get_size())*1.5, grid_comps=comp_grids, padding=parameters.padding)
        
#     # all_comp_grids.append(comp_grids)
#     # self.all_los_grids.append(los_grids)
#     # self.all_los.append(los)
#     # self.all_ol_grids.append(ol_grids)
#     # self.all_ol.append(ol)

#     dom, resultant, all_vecs = compute_pad_referenced_distance_vectors_v2(
#         parameters.node, 
#         parameters.neighbors, 
#         parameters.eoi,
#         ignore_power=parameters.ignore_power_nets
#         )

#     # group
#     mp, eucledian_dist, angle = compute_vector_to_group_midpoint(
#         parameters.node, 
#         parameters.neighbors
#     )            

#     if tracker is not None:
#         tracker.add_observation(comp_grids=comp_grids)
#         tracker.add_ratsnest(draw_ratsnest(parameters.node, parameters.neighbors, parameters.eoi, parameters.board_width, parameters.board_height, padding=parameters.padding, ignore_power=parameters.ignore_power_nets))

#     return { "los":  los[-8:],
#                     "ol": ol[-8:],
#                     "dom": [dom[0], dom[1]],
#                     "euc_dist": [ eucledian_dist, angle ],
#                     "position": [parameters.node.get_pos()[0] / parameters.board_width, parameters.node.get_pos()[1] / parameters.board_height],                    
#                     "ortientation": [wrap_angle(parameters.node.get_orientation())]
#         }        
