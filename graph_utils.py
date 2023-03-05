#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:07:44 2022

@author: luke
"""

import pcb as pcb
import graph.graph as graph
import graph.board as board
import graph.node as node
import graph.edge as edge
import logging 
import numpy as np

def extract_graph(pcb_file="", pcb_vec=[], remove_self_loops_in_multi_comp_nets=False):
    '''
    Parameters
    ----------
    pcb_file : TYPE
        path to a pcb file containing a design.

    Returns
    -------
    list
        Returns a list containing two lists. A list of nodes and a list of edges.
        
    Node position is translated such that the board has an origin at (0, 0)

    '''
    p = pcb.pcb()
    g = graph.graph()
    b = board.board()
    
    # Create empty vectors for nodes and edges
    nv = node.n_vec()
    ev = edge.e_vec()
    
    if (pcb_file != ""):       
        # Create a vector of pcb pointers to hold the list of pcbs to be read from a .pcb file.
        pv = pcb.vptr_pcbs()
                
        pcb.read_pcb_file(pcb_file,pv)      # Read pcb file
        p = pv[0]                           # load pcb 

    elif (len(pcb_vec) > 0):
        p = pcb_vec[0]
    
    else:
        logging.error(f"in file {__file__} in function {extract_graph.__name__}, either.pcb file or pcb object is required.")
        return -1
        
    p.get_graph(g)          # copy graph
    p.get_board(b)          # copy board
    
    g.set_component_origin_to_zero(b)       # set component origin to zero
    
    nv = g.get_nodes()
    ev = g.get_edges()
    
    G = []
    V = []
    E = []
    edge_features = []    
    
    # node features
    for n in nv:
        v = []
        
        size = n.get_size()
        v.append(float(size[0]))
        v.append(float(size[1]))
        
        pos = n.get_pos()
        v.append(float(pos[0]))
        v.append(float(pos[1]))
    
        v.append(float(n.get_orientation()))
        v.append(float(n.get_pin_count()))
        #n.print_to_console(True)
        #print(v)
        V.append(v)
        
    print('')
    
    # Omitting ground net only and pruning duplicate edges
    # duplicate = 0
    # # edges and edge features        
    # for e1 in ev:
    #     e = []
    #     duplicate_found = False
        
    #     # Omit GND edges only
    #     if (e1.get_power_rail() == 1):    
    #         continue
        
    #     e.append(int(e1.get_instance_id(0)))
    #     e.append(int(e1.get_instance_id(1)))
    #     #edge.print_to_console(True)
    #     #print(e)
        
    #     for e2 in E:
    #         if ((e[0] == e2[0] and e[1] == e2[1]) or (e[0] == e2[1] and e[1] == e2[0])):
    #             duplicate += 1
    #             duplicate_found = True
    #             break
            
    #     if not duplicate_found:
    #         E.append(e)
    #     #print('')
    
    # Omitting all power nets and summing duplicate edges in edge_features vector.
    duplicate = 0
    # edges and edge features        
    for e1 in ev:
        e = []
        duplicate_found = False
        # Omit all power edges
        if (e1.get_power_rail() > 0):
            continue
        
        # Omit GND edges only
        if (e1.get_power_rail() == 1):    
            continue
        
        # Other nets to omit
        if(e1.get_net_name()[-3:] == "vin" or e1.get_net_name() == "VDD" or e1.get_net_name() == "VCC" or e1.get_net_name() == "VDDA" or e1.get_net_name() == "GND1"):
            continue
               
        # Specific to BananaSchplit_w_connectors_and_mounting_hole_*.kicad_pcb
        if(e1.get_net_name() == "+12V" or e1.get_net_name() == "-12V"):
            continue
        
        # Specific to tc_logger_w_connectors_*.kicad_pcb
        # Nets 3V3, /power/5V0 USB_5V0 /power/VIN
        if(e1.get_net_name() == "3V3" or e1.get_net_name()[-3:] == "5V0" or e1.get_net_name()[-3:] == "VIN"):
            continue
        
        # Specific to quickfeather-board_connectors_only_*.kicad_pcb
        if(e1.get_net_name() == "+VBUS" or e1.get_net_name() == "+VBAT"):
            continue
        
        e.append(int(e1.get_instance_id(0)))
        e.append(int(e1.get_instance_id(1)))
        #edge.print_to_console(True)
        #print(e)
        if remove_self_loops_in_multi_comp_nets:
            if g.components_in_net(e1.get_net_name()) > 1:
                if e[0] == e[1]:
                    #print(f'Ignoring self loops in net {e1.get_net_name()}')
                    continue

        for e2, edge_feature in zip(E, edge_features):
            if ((e[0] == e2[0] and e[1] == e2[1]) or (e[0] == e2[1] and e[1] == e2[0])):
                edge_feature[0] += 1              # increment the number of edges
                edge_feature[1].append(e1.get_net_name())
                duplicate += 1
                duplicate_found = True
                break
            
        if not duplicate_found:
            edge_features.append([1,[e1.get_net_name()]])
            E.append(e)
        #print('')
    
    # for e, edge_feature in zip(E, edge_features):
    #     print(e, edge_feature)
    print(f'Vertices in the graph : {len(V)}')
    print(f'Edges in the graph    : {len(E)} exclusing {duplicate} duplicate edges')
    
    return [V,E,edge_features]

def extract_pad_graph(pcb_file="", pcb_vec=[],):   
    '''
    Parameters
    ----------
    pcb_file : TYPE
        path to a pcb file containing a design.

    Returns
    -------
    list
        Returns a list containing two lists. A list of nodes and a list of edges.
        The nodes represent component pads.
    Node position is translated such that the board has an origin at (0, 0)

    '''
    p = pcb.pcb()
    g = graph.graph()
    b = board.board()
    
    # Create empty vectors for nodes and edges
    nv = node.n_vec()
    ev = edge.e_vec()
    
    if (pcb_file != ""):       
        # Create a vector of pcb pointers to hold the list of pcbs to be read from a .pcb file.
        pv = pcb.vptr_pcbs()
                
        pcb.read_pcb_file(pcb_file,pv)      # Read pcb file
        p = pv[0]                           # load pcb 

    elif (len(pcb_vec) > 0):
        p = pcb_vec[0]
    
    else:
        logging.error(f"in file {__file__} in function {extract_pad_graph.__name__}, either.pcb file or pcb object is required.")
        return -1
        
    p.get_graph(g)          # copy graph
    p.get_board(b)          # copy board
    
    g.set_component_origin_to_zero(b)       # set component origin to zero
       
    nv = g.get_nodes()
    ev = g.get_edges()
    
    G = []
    V = []
    E = []
    
    duplicate = 0
    
    pads = []
    prev_pin_count = 0
    
    for n in nv:
        if len(pads) == 0:
            pads.append(0)
        else:
            pads.append(pads[-1] + prev_pin_count)
            
        prev_pin_count = n.get_pin_count()
        #n.print_to_console(True)
        #print(v)
        
    #print(pads)
    v_count = pads[-1] + prev_pin_count
    #print(v_count)
    for i in range(v_count):
        V.append([0.0,0.0,0.0,0.0])
    #print(V)
    print('')
        
    n = node.node()  
    src_pos = [0,0]
    dst_pos = [0,0]
    src_size = [0,0]
    dst_size = [0,0]
    
    for e1 in ev:
        e = []
        duplicate_found = False
        # Omit all power edges
        #if (edge.get_power_rail() > 0):
            #continue
        
        # Omit GND edges only
        # if (e1.get_power_rail() == 1):    
        #     continue
        
        # if (e1.get_power_rail() == 2):    
        #     continue
        
        # index 0 - id of node a
        # index 1 - pad id corresponding to node a
        # index 8 - id of node b
        # index 9 - pad id corresponding to node b        
        src = pads[e1.get_instance_id(0)] + e1.get_pad_id(0)
        dst = pads[e1.get_instance_id(1)] + e1.get_pad_id(1)
        
        e.append(src)
        e.append(dst)
        #edge.print_to_console(True)
        #print(e)
        
        for e2 in E:
            if ((e[0] == e2[0] and e[1] == e2[1]) or (e[0] == e2[1] and e[1] == e2[0])):
                duplicate += 1
                duplicate_found = True
                break
            
        if not duplicate_found:
            E.append(e)
            
            # index 0 - id of node a
            # index 1 - pad id corresponding to node a
            # index 8 - id of node b
            # index 9 - pad id corresponding to node b
            # src = pads[e1.get_instance_id(0)] + e1.get_pad_id(0)
            # dst = pads[e1.get_instance_id(1)] + e1.get_pad_id(1)
            
            g.get_node_by_id(e1.get_instance_id(0), n)
            vpos = n.get_pos()
            # vsize = n.get_size()
            # if (n.get_orientation() == 90.0) or (n.get_orientation() == 270.0):
            #     tmp = vsize[0]
            #     vsize[0] = vsize[1]
            #     visze[1] = tmp
                
            epos = e1.get_pos(0)
            #r_vpos = kicad_rotate(float(vpos[0]), float(vpos[1]), float(n.get_orientation()))
            r_epos = kicad_rotate(float(epos[0]), float(epos[1]), float(n.get_orientation()))
            src_pos[0] = vpos[0] + r_epos[0]
            src_pos[1] = vpos[1] + r_epos[1]

            esize = e1.get_size(0)
            #r_esize = kicad_rotate(float(esize[0]), float(esize[1]), float(n.get_orientation()))
            src_size[0] = abs(esize[0])
            src_size[1] = abs(esize[1])
            

            g.get_node_by_id(e1.get_instance_id(1), n)
            vpos = n.get_pos()
            epos = e1.get_pos(1)
            r_epos = kicad_rotate(float(epos[0]), float(epos[1]), float(n.get_orientation()))
    
            esize = e1.get_size(1)
            #r_esize = kicad_rotate(float(esize[0]), float(esize[1]), float(n.get_orientation()))
            dst_size[0] = abs(esize[0])
            dst_size[1] = abs(esize[1])
    
            dst_pos[0] = vpos[0] + r_epos[0]
            dst_pos[1] = vpos[1] + r_epos[1]
            #print(f'{e1.get_instance_id(0)}.{e1.get_pad_id(0)} -> {src}, {e1.get_instance_id(1)}.{e1.get_pad_id(1)} -> {dst}')
            #print(f'{round(src_pos[0],2)},{round(src_pos[1],2)} -> {round(dst_pos[0],2)},{round(dst_pos[1],2)}')
            
            if (V[src] != [0.0,0.0,0.0,0.0]) :
                if (V[src][2] != src_pos[0]) and (V[src][3] != src_pos[1]):
                    print(f'V[src] contains {[round(num,2) for num in V[src]]} and is being overwritten with {[round(num,2) for num in src_pos]}')
            else:
                V[src][0] = src_size[0]
                V[src][1] = src_size[1]
                V[src][2] = src_pos[0] 
                V[src][3] = src_pos[1]
                
            if (V[dst] != [0.0,0.0,0.0,0.0]) :
                if (V[dst][2] != dst_pos[0]) and (V[dst][3] != dst_pos[1]):
                    print(f'V[dst] contains {[round(num,2) for num in V[dst]]} and is being overwritten with {[round(num,2) for num in dst_pos]}')
            else:
                V[dst][0] = dst_size[0]
                V[dst][1] = dst_size[1]
                V[dst][2] = dst_pos[0]
                V[dst][3] = dst_pos[1]
            #print('')
        
    print(f'Vertices in the graph : {len(V)}')
    print(f'Edges in the graph    : {len(E)} exclusing {duplicate} duplicate edges')
    
    #print(V)
    # print(E)
    #print(len(V), E)
    
    i = 0
    V2 = []
    for vv in V:
        if (vv[0] == 0.0) and (vv[1] == 0.0):
            #V2.remove(vv)  
            
            for ee in E:
                for j in range(2):
                    if ee[j] > i:
                        ee[j] -= 1
        else:
            V2.append(vv)
            i += 1
                
    # G2 = nx.Graph()
    # i = 0
    # for vv in V3:
    #     G2.add_node(str(i))
    #     i+= 1
        
    # G2.add_edges_from([tuple([str(x[0]),str(x[1])]) for x in E2])
    # nx.draw_planar(G2, with_labels=True)
    # pos = nx.spring_layout(G2)
    # nx.draw_networkx_nodes(G2, pos, cmap=plt.get_cmap('jet'))
    # nx.draw_networkx_labels(G2, pos)
    # nx.draw_networkx_edges(G2, pos, [tuple([str(x[0]),str(x[1])]) for x in E2])
    
    print(f'Vertices in the graph : {len(V2)}')
    return [V2,E]

def kicad_rotate( x, y, a):

# 	 // --             --     -- --
# 	 // |  cos()   sin() | \/  | x |
# 	 // | -sin()   cos() | /\  | y |
# 	 // --             --     -- --
#   x -> x-coord
#   y -> y-coord
#   a -> angle in degrees
	theta =  np.pi * (a / 180.0)

	rx = x * np.cos( theta ) + y * np.sin( theta )
	ry = - x * np.sin( theta ) + y * np.cos( theta )

	return [rx,ry]

def kicad_rotate_around_point( x, y, cx, cy, a):

# 	 // --             --     -- --
# 	 // |  cos()   sin() | \/  | x |
# 	 // | -sin()   cos() | /\  | y |
# 	 // --             --     -- --
#   x -> x-coord
#   y -> y-coord
#   a -> angle in degrees
	theta =  np.pi * (a / 180.0)
	
	rx = ((x - cx) * np.cos( theta ) + ( y - cy ) * np.sin( theta )) + cx
	ry = (-(x - cx) * np.sin( theta ) + ( y - cy ) * np.cos( theta )) + cy

	return [rx,ry]
