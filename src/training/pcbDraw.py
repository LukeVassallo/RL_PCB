#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:38:05 2022

@author: luke
"""

import numpy as np
from PIL import Image
import cv2
import logging 
import graph.graph as graph
import graph.board as board
from graph_utils import kicad_rotate
import sys 

r = 0.1    # resolution in mm 
#r = 0.05    # resolution in mm (Use for evaluation only)
def draw_board(g_pad, g_comp, bx, by):
    x = bx / r;
    y = by / r;
    
    grid = []
    gridline = []
    for i in range(int(x)):
        gridline.append(0.0)
    
    for i in range(int(y)):
        grid.append(gridline)
        
    grid = np.array(grid)
    print(f'grid shape    : {grid.shape}')    
    
    if bool(g_pad):
        for v in g_pad[0]:
            print(f'size     : ({v[0]},{v[1]})')
            print(f'pos      : ({v[2]},{v[3]})')
            
            for i in range(int(x)):
                scaled_i = i * r
                if (scaled_i > (float(v[2]) - float(v[0]/2))) and (scaled_i < (float(v[2]) + float(v[0]/2))):
                   
                    for j in range(int(y)):
                        scaled_j = j * r
                        if (scaled_j > (float(v[3]) - float(v[1]/2))) and (scaled_j < (float(v[3]) + float(v[1]/2))):
                            grid[j,i] += 0.5
    
    print()                        
    for v in g_comp[0]:
        print(f'size     : ({v[0]},{v[1]})')
        print(f'pos      : ({v[2]},{v[3]})')
        

        if (v[4] == 90.0) or (v[4] == 270):
            vx = float(v[1]/2)
            vy = float(v[0]/2)
        else:
            vx = float(v[0]/2)
            vy = float(v[1]/2)            
        
        for i in range(int(x)):
            scaled_i = i * r
            if (scaled_i > (float(v[2]) - vx)) and (scaled_i < (float(v[2]) + vx)):
               
                for j in range(int(y)):
                    scaled_j = j * r
                    if (scaled_j > (float(v[3]) - vy)) and (scaled_j < (float(v[3]) + vy)):
                        grid[j,i] += 0.5

    print(grid.shape)
    scaled_grid = grid * 255.0
    im = Image.fromarray(scaled_grid.astype(np.uint8), 'L')
    im.show()
    # tcks = []

    # for i in range(0,int(len(grid)),int(len(grid)/5)):
    #     tcks.append(i)
    
    # tcks = np.array(tcks, dtype='float32')

    # plt.xlabel('x (μm)')
    # plt.ylabel('y (μm)')
    # plt.xticks(tcks)
    # plt.yticks(tcks)
    # print(tcks)

    # #plt.figure(figsize=(10,6))
    # plt.imshow(grid)
    # #plt.colorbar(cmap='gray')
    # plt.show()
    #plt.close()   

    return grid       

# Uses open cv to draw the board's components and their pads
def draw_board2(g_pad, g_comp, bx, by, padding=None):
    x = bx / r;
    y = by / r;
    
    if padding is not None:
        grid_comps = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
        grid_pads = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)

    else:
        grid_comps = np.zeros((int(x),int(y),1), np.uint8)
        grid_pads = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)

    
    # grid_comps = np.zeros((int(x),int(y),1), np.uint8)
    # grid_pads = np.zeros((int(x),int(y),1), np.uint8)

    # print(f'grid_comps shape    : {grid_comps.shape}')    
    # print(f'grid_pads shape    : {grid_pads.shape}')    

    if bool(g_pad):
        for v in g_pad[0]:
            # print(f'size     : ({v[0]},{v[1]})')
            # print(f'pos      : ({v[2]},{v[3]})')
            
            if padding is not None:
                xc = float(v[2]) / r + int(padding/r) 
                yc = float(v[3]) / r + int(padding/r)
            else:
                xc = float(v[2]) / r 
                yc = float(v[3]) / r 
                
            sz_x = float(v[0]) / r
            sz_y = float(v[1]) / r
            box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), 0)) # convert the center, size and orientation to rectange points
            box = np.int0(box)  # ensure that box point are integers
            cv2.drawContours(grid_pads,[box],0,(64),-1)
    
    for v in g_comp[0]:
        # print(f'size     : ({v[0]},{v[1]})')
        # print(f'pos      : ({v[2]},{v[3]})')
        
        if (v[4] == 90.0) or (v[4] == 270):
            vx = float(v[1]/2)
            vy = float(v[0]/2)
        else:
            vx = float(v[0]/2)
            vy = float(v[1]/2)   
   
        # method #1 - Draw a rectangle with two points.            
        # xmin = int( (float(v[2]) - vx) / r )
        # xmax = int( (float(v[2]) + vx) / r )
        # ymin = int( (float(v[3]) - vy) / r )
        # ymax = int( (float(v[3]) + vy) / r )        
        # cv2.rectangle(grid, (xmin,ymin), (xmax,ymax), (32), -1)
        
        # method #2 - Draw a rectange with center, size and angle.
        if padding is not None:
            xc = float(v[2]) / r + int(padding/r) 
            yc = float(v[3]) / r + int(padding/r)
        else:
            xc = float(v[2]) / r 
            yc = float(v[3]) / r 
                
        sz_x = float(v[0]) / r
        sz_y = float(v[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -v[4])) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        cv2.drawContours(grid_comps,[box],0,(64),-1)



    
    cv2.imshow("board layout", grid_comps+grid_pads)
    
    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_comps, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border,
              cv2.copyMakeBorder(grid_pads, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))]
    else:
        return grid_comps, grid_pads
    
def draw_board_from_board_and_graph(b, g, draw_placed=True, draw_unplaced=True, padding=None, line_thickness=-1): 
    """
    

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    draw_placed : TYPE, optional
        DESCRIPTION. The default is True.
    draw_unplaced : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    grid_comps : n-dimensional numpy array
        Contains two grid with grayscale drawings. In position 0 there is a drawing containing 
        the board with the placed components. Position 1 contains the unplaced component.
        
    

    """       
    nv = g.get_nodes()
    ev = g.get_edges()
    
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        grid_comps = np.zeros((2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)
    padded_grid_comps = []
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
                                 
    for n in nv:       
        # if (n.get_isPlaced() == 1):
        #     if (draw_placed == False): continue
        # else:
        #     if (draw_unplaced == False): continue
        
        pos = n.get_pos()
        size = n.get_size()
        orientation = n.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (n.get_isPlaced() == 1):
            cv2.drawContours(grid_comps[0],[box],0,(64),line_thickness)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),line_thickness)
                
    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border,
              cv2.copyMakeBorder(grid_comps[1], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))]
    else:
        return grid_comps
 
def draw_board_from_board_and_graph_with_debug(b, g, draw_placed=True, draw_unplaced=True, padding=None, line_thickness=-1): 
    """
    

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    draw_placed : TYPE, optional
        DESCRIPTION. The default is True.
    draw_unplaced : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    grid_comps : n-dimensional numpy array
        Contains two grid with grayscale drawings. In position 0 there is a drawing containing 
        the board with the placed components. Position 1 contains the unplaced component.
        
    

    """       
    nv = g.get_nodes()
    ev = g.get_edges()
    
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        grid_comps = np.zeros((3,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((3,int(x),int(y),1), np.uint8)
    padded_grid_comps = []
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
                                 
    for n in nv:       
        # if (n.get_isPlaced() == 1):
        #     if (draw_placed == False): continue
        # else:
        #     if (draw_unplaced == False): continue
        
        pos = n.get_pos()
        size = n.get_size()
        orientation = n.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (n.get_isPlaced() == 1):
            cv2.drawContours(grid_comps[0],[box],0,(64),line_thickness)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),line_thickness)
            
        if padding is not None:
            tmp = draw_node_name(n, b.get_width(), b.get_height(), padding=padding)
            tmp = np.reshape(tmp,(tmp.shape[0],tmp.shape[1],1))
            grid_comps[2] = np.maximum(tmp, grid_comps[2])
                
    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border,
              cv2.copyMakeBorder(grid_comps[1], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)),
              cv2.copyMakeBorder(grid_comps[2], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))]
    else:
        return grid_comps           

    # cv2.imshow("board layout - placed", grid_comps[0])
    # #cv2.waitKey(500)
    # cv2.imshow("board layout - unplaced", grid_comps[1])
    
def draw_board_from_board_and_graph_multi_agent(b, g, node_id, draw_placed=True, draw_unplaced=True, padding=None, line_thickness=-1): 
    """
    

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    draw_placed : TYPE, optional
        DESCRIPTION. The default is True.
    draw_unplaced : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    grid_comps : n-dimensional numpy array
        Contains two grid with grayscale drawings. In position 0 there is a drawing containing 
        the board with the placed components. Position 1 contains the unplaced component.
        
    

    """       
    nv = g.get_nodes()
    ev = g.get_edges()
    
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        grid_comps = np.zeros((len(nv)+1,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((len(nv)+1,int(x),int(y),1), np.uint8)
    padded_grid_comps = []
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
    idx = 2                                 
    for n in nv:       
        # if (n.get_isPlaced() == 1):
        #     if (draw_placed == False): continue
        # else:
        #     if (draw_unplaced == False): continue
        
        pos = n.get_pos()
        size = n.get_size()
        orientation = n.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (n.get_id() == node_id):
            cv2.drawContours(grid_comps[1],[box],0,(64),line_thickness)
        else:
            cv2.drawContours(grid_comps[idx],[box],0,(64),line_thickness)
            idx +=1

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        stack = []
        
        stack.append(cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border)
        for i in range(1, idx, 1):
            stack.append(cv2.copyMakeBorder(grid_comps[i], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)))
        return stack
    else:
        print('draw_board_from_board_and_graph_multi_agent requires padding.')
        sys.exit()

def draw_comps_from_nodes_and_edges(n, nn, e, b, padding=None):   
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        grid_comps = np.zeros((2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
    
    # draw current node
    pos = n.get_pos()
    size = n.get_size()
    orientation = n.get_orientation()
    
    if padding is not None:
        xc = float(pos[0]) / r + int(padding/r)
        yc = float(pos[1]) / r + int(padding/r)
    else:
        xc = float(pos[0]) / r 
        yc = float(pos[1]) / r 
        
    sz_x = float(size[0]) / r
    sz_y = float(size[1]) / r
    box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
    box = np.int0(box)  # ensure that box point are integers
    if (n.get_isPlaced() == 1): # should always return False!
        cv2.drawContours(grid_comps[0],[box],0,(64),-1)
    else:
        cv2.drawContours(grid_comps[1],[box],0,(64),-1)
                   
    # draw neighbor nodes
    for v in nn:        
        pos = v.get_pos()
        size = v.get_size()
        orientation = v.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (v.get_isPlaced() == 1):  # should always return True!
            cv2.drawContours(grid_comps[0],[box],0,(64),-1)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),-1)

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border,
              cv2.copyMakeBorder(grid_comps[1], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))]
    else:
        return grid_comps

# identical to "draw_comps_from_nodes_and_edges(n, nn, e, b, padding=None)". This one makes use of a different interface.
def draw_board_from_nodes_and_edges(n, nn, e, bx, by, padding=None):   
    # Setup grid
    x = bx / r;
    y = by / r;
    
    if padding is not None:
        grid_comps = np.zeros((2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
    
    # draw current node
    pos = n.get_pos()
    size = n.get_size()
    orientation = n.get_orientation()
    
    if padding is not None:
        xc = float(pos[0]) / r + int(padding/r)
        yc = float(pos[1]) / r + int(padding/r)
    else:
        xc = float(pos[0]) / r 
        yc = float(pos[1]) / r 
        
    sz_x = float(size[0]) / r
    sz_y = float(size[1]) / r
    box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
    box = np.int0(box)  # ensure that box point are integers
    if (n.get_isPlaced() == 1): # should always return False!
        cv2.drawContours(grid_comps[0],[box],0,(64),-1)
    else:
        cv2.drawContours(grid_comps[1],[box],0,(64),-1)
                   
    # draw neighbor nodes
    for v in nn:        
        pos = v.get_pos()
        size = v.get_size()
        orientation = v.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (v.get_isPlaced() == 1):  # should always return True!
            cv2.drawContours(grid_comps[0],[box],0,(64),-1)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),-1)

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border,
              cv2.copyMakeBorder(grid_comps[1], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))]
    else:
        return grid_comps    
   
# The following is the difference between this and the prior method   
# neighbor nodes are always treated as 'locked' 
# current node is always treated as 'unlocked'
def draw_board_from_nodes_and_edges_multi_agent(n, nn, e, bx, by, padding=None):   
    # Setup grid
    x = bx / r;
    y = by / r;
    
    if padding is not None:
        grid_comps = np.zeros((len(nn)+2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((len(nn)+2,int(x),int(y),1), np.uint8)
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
    
    # draw current node
    pos = n.get_pos()
    size = n.get_size()
    orientation = n.get_orientation()
    
    if padding is not None:
        xc = float(pos[0]) / r + int(padding/r)
        yc = float(pos[1]) / r + int(padding/r)
    else:
        xc = float(pos[0]) / r 
        yc = float(pos[1]) / r 
        
    sz_x = float(size[0]) / r
    sz_y = float(size[1]) / r
    box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
    box = np.int0(box)  # ensure that box point are integers
    cv2.drawContours(grid_comps[1],[box],0,(64),-1)

    idx =2                 
    # draw neighbor nodes
    for v in nn:        
        pos = v.get_pos()
        size = v.get_size()
        orientation = v.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        cv2.drawContours(grid_comps[idx],[box],0,(64),-1)
        idx += 1


    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        tmp =[]
        for i in range(len(nn)+2):
            if i == 0:
                tmp.append(cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border)
            else:
                tmp.append(cv2.copyMakeBorder(grid_comps[i], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)))
                
        return tmp
    else:
        return grid_comps

# idx = 0 ( grid border )
# idx = 1 ( current node  )
# idx = 2 ... ( neighbors ... )

def draw_board_from_graph_multi_agent(g, node_id, bx, by, padding=None):   
    # Setup grid
    x = bx / r;
    y = by / r;
    
    all_nodes = g.get_nodes()    
    
    if padding is not None:
        grid_comps = np.zeros((len(all_nodes)+1,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((len(all_nodes)+1,int(x),int(y),1), np.uint8)
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
    
    # draw current node
    # pos = n.get_pos()
    # size = n.get_size()
    # orientation = n.get_orientation()
    
    # if padding is not None:
    #     xc = float(pos[0]) / r + int(padding/r)
    #     yc = float(pos[1]) / r + int(padding/r)
    # else:
    #     xc = float(pos[0]) / r 
    #     yc = float(pos[1]) / r 
        
    # sz_x = float(size[0]) / r
    # sz_y = float(size[1]) / r
    # box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
    # box = np.int0(box)  # ensure that box point are integers
    # cv2.drawContours(grid_comps[1],[box],0,(64),-1)

    idx =2                 
    # draw neighbor nodes
    for i in range(len(all_nodes)):        
        pos = all_nodes[i].get_pos()
        size = all_nodes[i].get_size()
        orientation = all_nodes[i].get_orientation()
        current_node_id = all_nodes[i].get_id()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (current_node_id == node_id):
            cv2.drawContours(grid_comps[1],[box],0,(64),-1)
        else:
            cv2.drawContours(grid_comps[idx],[box],0,(64),-1)
            idx += 1

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        tmp =[]
        for i in range(len(all_nodes)+1):
            if i == 0:
                tmp.append(cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border)
            else:
                tmp.append(cv2.copyMakeBorder(grid_comps[i], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)))
                
        return tmp
    else:
        return grid_comps        


# only comp_grids[0] is used.    
def draw_board_from_nodes_multi_agent(n, bx, by, padding=None):   
    # Setup grid
    x = bx / r;
    y = by / r;
    
    if padding is not None:
        grid_comps = np.zeros((2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)
    grid_pads = np.zeros((int(x),int(y),1), np.uint8)
                       
    # draw nodes
    for v in n:        
        pos = v.get_pos()
        size = v.get_size()
        orientation = v.get_orientation()
        
        if padding is not None:
            xc = float(pos[0]) / r + int(padding/r)
            yc = float(pos[1]) / r + int(padding/r)
        else:
            xc = float(pos[0]) / r 
            yc = float(pos[1]) / r 
            
        sz_x = float(size[0]) / r
        sz_y = float(size[1]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        cv2.drawContours(grid_comps[0],[box],0,(64),-1)


    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border, int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64))
       
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_comps[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(64))+border,
              cv2.copyMakeBorder(grid_comps[1], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))]
    else:
        return grid_comps        
    
def draw_ratsnest_with_board(current_node, neighbor_nodes, e, b, line_thickness=1, padding=None, ignore_power=False):
    # Setup grid
    bx = b.get_width()
    by = b.get_height()
    
    return draw_ratsnest(current_node, neighbor_nodes, e, bx, by, line_thickness=line_thickness, padding=padding, ignore_power=ignore_power)

    
    
    
#def draw_ratsnest(current_node, neighbor_nodes, e, b, line_thickness=1, padding=None, ignore_power=False):
    # Setup grid
   # x = b.get_width() / r;
   # y = b.get_height() / r;
   
def draw_ratsnest(current_node, neighbor_nodes, e, bx, by, line_thickness=1, padding=None, ignore_power=False):    
    x = bx / r
    y = by / r
    
    if padding is not None:
        ratsnest = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        ratsnest = np.zeros((int(x),int(y),1), np.uint8)
        
    # draw current node
    current_node_id = current_node.get_id()
    current_node_pos = current_node.get_pos()
    current_node_size = current_node.get_size()
    current_node_orientation = current_node.get_orientation()

    src = []
    dst = []
    for ee in e:
        if (ignore_power == True) and (ee.get_power_rail() > 0): 
            continue
        # Get list of pos, size for all 
        for i in range(2):
            if ee.get_instance_id(i) == current_node_id:
                pad_pos = ee.get_pos(i)
                rotated_pad_pos = kicad_rotate(float(pad_pos[0]),float(pad_pos[1]), current_node_orientation) # rotate pad positions so that they match the component's orientation
                src.append([current_node_pos[0] + rotated_pad_pos[0],
                            current_node_pos[1] + rotated_pad_pos[1]
                            ])
            else:
                for n in neighbor_nodes:
                    if ee.get_instance_id(i) == n.get_id():
                        neighbor_node_pos = n.get_pos()
                        pad_pos = ee.get_pos(i)
                        rotated_pad_pos = kicad_rotate(float(pad_pos[0]),float(pad_pos[1]), n.get_orientation()) # rotate pad positions so that they match the component's orientation

                        dst.append([neighbor_node_pos[0] + rotated_pad_pos[0],
                                    neighbor_node_pos[1] + rotated_pad_pos[1]
                                    ])
                        break;
                        
    for i in range(len(src)):
        if padding is not None:
            sx = int(src[i][0]/r) + int(padding/r)
            sy = int(src[i][1]/r) + int(padding/r)
            
            dx = int(dst[i][0]/r) + int(padding/r)
            dy = int(dst[i][1]/r) + int(padding/r)
        else:
            sx = int(src[i][0]/r) 
            sy = int(src[i][1]/r)
            
            dx = int(dst[i][0]/r)
            dy = int(dst[i][1]/r)
        
        cv2.line(ratsnest, (sx,sy), (dx,dy), (255), line_thickness) # image, pt1 (x,y), pt2, color (BGR), thickness
    
    if padding is not None:
        return cv2.copyMakeBorder(ratsnest, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:
        return ratsnest

def draw_pads_from_nodes_and_edges(n, nn, e, b, padding=None, ratsnest=False):   
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        grid_pads = np.zeros((2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
        grid_lines = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_pads = np.zeros((2,int(x),int(y),1), np.uint8)
        grid_lines = np.zeros((2,int(x),int(y),1), np.uint8)
    
    # draw current node
    node_pos = n.get_pos()
    node_size = n.get_size()
    node_orientation = n.get_orientation()

    current_pads = []
    current_pads_wo_duplicates = []
    for ee in e:
        # Get list of pos, size for all 
        for i in range(2):
            if ee.get_instance_id(i) == n.get_id():
                pad_pos = ee.get_pos(i)
                rotated_pad_pos = kicad_rotate(float(pad_pos[0]),float(pad_pos[1]), node_orientation) # rotate pad positions so that they match the component's orientation
                pad_size = ee.get_size(i)
                current_pads.append([ node_pos[0] + rotated_pad_pos[0],
                                     node_pos[1] + rotated_pad_pos[1],
                                     pad_size[0],
                                     pad_size[1] ])
        
    # drop duplicates
    for i in current_pads:
        if i not in current_pads_wo_duplicates:
            current_pads_wo_duplicates.append(i)
            
    
    # plot   
    for pad in current_pads_wo_duplicates:
        if padding is not None:
            xc = float(pad[0]) / r + int(padding/r)
            yc = float(pad[1]) / r + int(padding/r)
        else:
            xc = float(pad[0]) / r 
            yc = float(pad[1]) / r 
            
        sz_x = float(pad[2]) / r
        sz_y = float(pad[3]) / r
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -node_orientation)) # convert the center, size and orientation to rectange points
        box = np.int0(box)  # ensure that box point are integers
        if (n.get_isPlaced() == 1): # should always return False!
            cv2.drawContours(grid_pads[0],[box],0,(64),-1)
        else:
            cv2.drawContours(grid_pads[1],[box],0,(64),-1)
   
    # draw neighbor nodes
    for v in nn:    
        node_pos = v.get_pos()
        node_size = v.get_size()
        node_orientation = v.get_orientation()
        
        neighbor_pads = []
        neighbor_pads_wo_duplicates = []
        for ee in e:
            # Get list of pos, size for all 
            for i in range(2):
                if ee.get_instance_id(i) == v.get_id():
                    pad_pos = ee.get_pos(i)
                    rotated_pad_pos = kicad_rotate(float(pad_pos[0]),float(pad_pos[1]), node_orientation) # rotate pad positions so that they match the component's orientation
                    pad_size = ee.get_size(i)
                    neighbor_pads.append([ node_pos[0] + rotated_pad_pos[0],
                                         node_pos[1] + rotated_pad_pos[1],
                                         pad_size[0],
                                         pad_size[1] ])
                    
        # drop duplicates
        for i in neighbor_pads:
            if i not in neighbor_pads_wo_duplicates:
                neighbor_pads_wo_duplicates.append(i)
        
        # plot   
        for pad in neighbor_pads_wo_duplicates:
            if padding is not None:
                xc = float(pad[0]) / r + int(padding/r)
                yc = float(pad[1]) / r + int(padding/r)
            else:
                xc = float(pad[0]) / r 
                yc = float(pad[1]) / r 
                
            sz_x = float(pad[2]) / r
            sz_y = float(pad[3]) / r
            box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -node_orientation)) # convert the center, size and orientation to rectange points
            box = np.int0(box)  # ensure that box point are integers
            if (v.get_isPlaced() == 1): # should always return False!
                cv2.drawContours(grid_pads[0],[box],0,(64),-1)
            else:
                cv2.drawContours(grid_pads[1],[box],0,(64),-1)
    
    if ratsnest == True:
        ratsnest_grid = draw_ratsnest(n, nn, e, b, padding)
    else:
        ratsnest_grid = None

    if padding is not None:
        # return [cv2.copyMakeBorder(grid_comps[0], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(64)),
        #      cv2.copyMakeBorder(grid_comps[1], int(padding/r), int(padding/r), int(padding/r), int(padding/r), cv2.BORDER_CONSTANT, value=(0))]
        return [cv2.copyMakeBorder(grid_pads[0], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)),
              cv2.copyMakeBorder(grid_pads[1], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))], ratsnest_grid 
    else:
        return grid_pads, ratsnest_grid
            

def draw_board3(p, draw_placed=True, draw_unplaced=True):
    """ 
    Parameters
    ----------
    p : TYPE
        pcb object.

    Returns
    -------
    grid.


    Draws the board using components only. It simply extracts the graph and 
    board objects than calls the draw_board_from_board_and_graph function that 
    does the actual heavy lifting.
    """
      
    g = graph.graph()
    b = board.board()
    
    p.get_graph(g)          # copy graph
    p.get_board(b)          # copy board
        
    return draw_board_from_board_and_graph(b, g, draw_placed, draw_unplaced)    
    
def draw_los(pos_x, pos_y, radius, angle_offset, bx, by, padding=None):
    """
    

    Parameters
    ----------
    pos_x : TYPE
        floating point node x-co-ordinate. This show not be formatted into pixels
    pos_y : TYPE
        floating point node y-co-ordinate. This show not be formatted into pixels
    radius : TYPE
        DESCRIPTION.
    bx : TYPE
        DESCRIPTION.
    by : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    x = bx / r;
    y = by / r;
    if padding is not None:
        los_segments = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        los_segments = np.zeros((8,int(x),int(y),1), np.uint8)

    padded_los_segments = []
    segment_pixels = np.zeros(8)
    if padding is not None:
        scaled_x = np.int0(pos_x / r) + int(padding/r)
        scaled_y = np.int0(pos_y / r) + int(padding/r)
    else:
        scaled_x = np.int0(pos_x / r)
        scaled_y = np.int0(pos_y / r)
    scaled_radius = np.int0(radius / r)
    logging.debug(f'scaled_x : {scaled_x}, scaled_y : {scaled_y}, scaled_radius : {scaled_radius}')
    #start = -22.5 - 90 + angle_offset
    #stop = 22.5 - 90 + angle_offset
    start = -22.5 - angle_offset
    stop = 22.5 - angle_offset
    for i in range(8):
        cv2.ellipse(los_segments[i], (scaled_x,scaled_y), (scaled_radius,scaled_radius), 0, start, stop, (16), -1) 
        segment_pixels[i] = np.sum(los_segments[i]) / 16
        start -= 45
        stop -= 45
        
    # for i in range(8):
    #     cv2.imshow("los", los_segments[i])
    #     cv2.waitKey(500)
   
    # this is dumb, but needed so arrays are matching. copyMakeBorder reshapes the output image.
    if padding is not None:
        for i in range(8):
            padded_los_segments.append(cv2.copyMakeBorder(los_segments[i], 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)))
            
        return np.array(padded_los_segments), segment_pixels 
    else:   
        return los_segments, segment_pixels

def draw_los2(pos_x, pos_y, radius, angle, angle_offset, bx, by, padding=None):
    x = bx / r;
    y = by / r;
    if padding is not None:
        los_segment = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        los_segment = np.zeros((int(x),int(y),1), np.uint8)

    segment_pixels = np.zeros(1)
    if padding is not None:
        scaled_x = np.int0(pos_x / r) + int(padding/r)
        scaled_y = np.int0(pos_y / r) + int(padding/r)
    else:
        scaled_x = np.int0(pos_x / r)
        scaled_y = np.int0(pos_y / r)
    scaled_radius = np.int0(radius / r)
    logging.debug(f'scaled_x : {scaled_x}, scaled_y : {scaled_y}, scaled_radius : {scaled_radius}')
    start = -angle/2 - 90 + angle_offset
    stop = angle/2 - 90 + angle_offset

    cv2.ellipse(los_segment, (scaled_x,scaled_y), (scaled_radius,scaled_radius), 0, start, stop, (16), -1) 
    segment_pixels = np.sum(los_segment) / 16

        
    # for i in range(8):
    #     cv2.imshow("los", los_segments[i])
    #     cv2.waitKey(500)
   
    # this is dumb, but needed so arrays are matching. copyMakeBorder reshapes the output image.
    if padding is not None:
        return cv2.copyMakeBorder(los_segment, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0)), segment_pixels         
    else:   
        return los_segment, segment_pixels

def draw_debug(n, bx, by, padding=None):
    """
    

    Parameters
    ----------
    n : node object
        DESCRIPTION.
    bx : TYPE
        DESCRIPTION.
    by : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Test for verifying rotation angle
    # grid = np.zeros((int(bx/r), int(by/r), 1), np.uint8)
    # for i in range(360):
    #     pos_x = n.get_pos()[0]
    #     pos_y = n.get_pos()[1]
    #     size_y = n.get_size()[1]
    #     p1 = (int(pos_x/r), int(pos_y/r))   
    #     p12 = np.array(kicad_rotate(0, size_y*-1.2, -i))  # function returns a list
    #     p2 = (int((pos_x+p12[0])/r), int((pos_y+p12[1])/r))
        

    #     cv2.line(grid, p1, p2, (64), 2)
    #     cv2.imshow('debug_grid', grid)
    #     cv2.waitKey(10)
        
    x = bx / r;
    y = by / r;
    if padding is not None:
        grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid = np.zeros((int(x),int(y),1), np.uint8)
    
    pos_x = n.get_pos()[0]
    pos_y = n.get_pos()[1]
    size_x = n.get_size()[0]/2
    p12 = np.array(kicad_rotate(size_x*1.2, 0, n.get_orientation()))  # function returns a list
    if padding == None:
        p1 = (int(pos_x/r), int(pos_y/r))  
        p2 = ( int((pos_x+p12[0])/r),
              int((pos_y+p12[1])/r) )
    else:
        p1 = ( int(pos_x/r) + int(padding/r), 
              int(pos_y/r) + int(padding/r) )
    
        p2 = ( int((pos_x+p12[0])/r) + int(padding/r),
              int((pos_y+p12[1])/r) + int(padding/r))

    

    # place orientation on the top right 
    if padding == None:
        text_origin = (int( (n.get_pos()[0] + n.get_size()[0]/2)/r ),
                       int( (n.get_pos()[1] - n.get_size()[1]/2)/r ))
    else:
        text_origin = (int( (n.get_pos()[0] + n.get_size()[0]/2)/r ) + int(padding/r),
                       int( (n.get_pos()[1] - n.get_size()[1]/2)/r ) + int(padding/r))
            
    cv2.putText(img=grid,
                text=f'{n.get_orientation()}',
                org=text_origin,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(128))    
    # print(p1)
    # print(p2)
    cv2.line(grid, p1, p2, (255), 2)
    # cv2.imshow('debug_grid', grid)
    
    # this is dumb, but needed so arrays are matching. copyMakeBorder reshapes the output image.
    if padding is not None:
        return cv2.copyMakeBorder(grid, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:   
        return grid
    
def draw_node_name(n, bx, by, padding=None, loc="top_right", designator_only=False):
    """
    

    Parameters
    ----------
    n : node object
        DESCRIPTION.
    bx : TYPE
        DESCRIPTION.
    by : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
        
    x = bx / r;
    y = by / r;
    if padding is not None:
        grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid = np.zeros((int(x),int(y),1), np.uint8)
    
    pos_x = n.get_pos()[0]
    pos_y = n.get_pos()[1]
    size_x = n.get_size()[0]/2 


    if loc == "top_left":
        if padding == None:
            text_origin = (int( (n.get_pos()[0] - n.get_size()[0]/2- 1)/r ),                        # - 1 since text moves from left to right
                        int( (n.get_pos()[1] - n.get_size()[1]/2 - 0.25 )/r ))
        else:
            text_origin = (int( (n.get_pos()[0] - n.get_size()[0]/2 - 1)/r ) + int(padding/r),      # - 1 since text moves from left to right
                        int( (n.get_pos()[1] - n.get_size()[1]/2 - 0.25)/r ) + int(padding/r))
    else:# loc == "top_right":
        # place orientation on the top right 
        if padding == None:
            text_origin = (int( (n.get_pos()[0] + n.get_size()[0]/2)/r ),
                        int( (n.get_pos()[1] - n.get_size()[1]/2)/r ))
        else:
            text_origin = (int( (n.get_pos()[0] + n.get_size()[0]/2)/r ) + int(padding/r),
                        int( (n.get_pos()[1] - n.get_size()[1]/2)/r ) + int(padding/r))

    if designator_only == True:
        cv2.putText(img=grid,
            text=f'{n.get_name()}',
            org=text_origin,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(127))    
    else:
        cv2.putText(img=grid,
                    text=f'{n.get_id()} ({n.get_name()})',
                    org=text_origin,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(127))    

    # this is dumb, but needed so arrays are matching. copyMakeBorder reshapes the output image.
    if padding is not None:
        return cv2.copyMakeBorder(grid, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:   
        return grid    
   
def draw_dom_vectors(vex, current_node, b, resultant=None, all_vecs=None, padding=None):
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        vex_grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        vex_grid = np.zeros((int(x),int(y),1), np.uint8)
        
    # current node position
    current_node_pos = current_node.get_pos()
    
    p1 = [current_node_pos[0], current_node_pos[1]]
    p2 = [0,0]
                            
    if padding is not None:        
        p2[0] = int((p1[0] + vex[0]*np.cos(vex[1]))/r) + int(padding/r)
        p2[1] = int((p1[1] - vex[0]*np.sin(vex[1]))/r) + int(padding/r)
        
        p1[0] = int(p1[0]/r) + int(padding/r)
        p1[1] = int(p1[1]/r) + int(padding/r)
    else:       
        p2[0] = int((p1[0] + vex[0]*np.cos(vex[1]))/r)
        p2[1] = int((p1[1] - vex[0]*np.sin(vex[1]))/r)
        
        p1[0] = int(p1[0]/r)
        p1[1] = int(p1[1]/r)
    
    cv2.line(vex_grid, tuple(p1), tuple(p2), (255), 1) # image, pt1 (x,y), pt2, color (BGR), thickness
    
    if resultant is not None:
        for res in resultant:
            #print(res)
            vec = res[-1]
            current_node_pos = current_node.get_pos()

            p1 = [current_node_pos[0], current_node_pos[1]]
            #p1 = [res[-3], res[-2]]
            p2 = [0,0]
                                    
            if padding is not None:        
                p2[0] = int((p1[0] + vec[0]*np.cos(vec[1]))/r) + int(padding/r)
                p2[1] = int((p1[1] - vec[0]*np.sin(vec[1]))/r) + int(padding/r)
                
                p1[0] = int(p1[0]/r) + int(padding/r)
                p1[1] = int(p1[1]/r) + int(padding/r)
            else:       
                p2[0] = int((p1[0] + vec[0]*np.cos(vec[1]))/r)
                p2[1] = int((p1[1] - vec[0]*np.sin(vec[1]))/r)
                
                p1[0] = int(p1[0]/r)
                p1[1] = int(p1[1]/r)   
                
            # print(p1,p2)
            cv2.line(vex_grid, tuple(p1), tuple(p2), (192), 2) # image, pt1 (x,y), pt2, color (BGR), thickness
            
    if all_vecs is not None:
        for vecs in all_vecs:
            for v in vecs:
                vec = [v[-2],v[-1]]
                p1 = [v[5], v[6]]
                p2 = [0,0]
                                        
                if padding is not None:        
                    p2[0] = int((p1[0] + vec[0]*np.cos(vec[1]))/r) + int(padding/r)
                    p2[1] = int((p1[1] - vec[0]*np.sin(vec[1]))/r) + int(padding/r)
                    
                    p1[0] = int(p1[0]/r) + int(padding/r)
                    p1[1] = int(p1[1]/r) + int(padding/r)
                else:       
                    p2[0] = int((p1[0] + vec[0]*np.cos(vec[1]))/r)
                    p2[1] = int((p1[1] - vec[0]*np.sin(vec[1]))/r)
                    
                    p1[0] = int(p1[0]/r)
                    p1[1] = int(p1[1]/r)   
                    
                cv2.line(vex_grid, tuple(p1), tuple(p2), (128), 1) # image, pt1 (x,y), pt2, color (BGR), thickness

    
    if padding is not None:
        return cv2.copyMakeBorder(vex_grid, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:
        return vex_grid
    
def draw_vector_to_group_midpoint(n, midpoint, vector, b, padding=None):
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        vex_grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        vex_grid = np.zeros((int(x),int(y),1), np.uint8)
        
    mp = [0,0]
    mp[0] = int(midpoint[0] / r)
    mp[1] = int(midpoint[1] / r)
        
    if padding is not None:
        mp[0] += int(padding/r)
        mp[1] += int(padding/r)    
             
    #     p2[0] = int((p1[0] + vec[0]*np.cos(vec[1]))/r) + int(padding/r)
    #     p2[1] = int((p1[1] - vec[0]*np.sin(vec[1]))/r) + int(padding/r)
        
    #     p1[0] = int(p1[0]/r) + int(padding/r)
    #     p1[1] = int(p1[1]/r) + int(padding/r)
    # else:       
    #     p2[0] = int((p1[0] + vec[0]*np.cos(vec[1]))/r)
    #     p2[1] = int((p1[1] - vec[0]*np.sin(vec[1]))/r)
        
    #     p1[0] = int(p1[0]/r)
    #     p1[1] = int(p1[1]/r)     
    cv2.circle(vex_grid, (mp[0],mp[1]), 3, (255), -1)  
    
    current_node_pos = n.get_pos()
    
    p1 = [current_node_pos[0], current_node_pos[1]]
    p2 = [0,0]
                            
    if padding is not None:        
        p2[0] = int((p1[0] + vector[0]*np.cos(vector[1]))/r) + int(padding/r)
        p2[1] = int((p1[1] - vector[0]*np.sin(vector[1]))/r) + int(padding/r)
        
        p1[0] = int(p1[0]/r) + int(padding/r)
        p1[1] = int(p1[1]/r) + int(padding/r)
    else:       
        p2[0] = int((p1[0] + vector[0]*np.cos(vector[1]))/r)
        p2[1] = int((p1[1] - vector[0]*np.sin(vector[1]))/r)
        
        p1[0] = int(p1[0]/r)
        p1[1] = int(p1[1]/r)   
        
    cv2.line(vex_grid, tuple(p1), tuple(p2), (255), 1) # image, pt1 (x,y), pt2, color (BGR), thickness
    
    if padding is not None:
        return cv2.copyMakeBorder(vex_grid, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:
        return vex_grid

def pcbDraw_resolution(): return r

def set_pcbDraw_resolution(resolution):
    global r
    r = resolution


def draw_current_node_center( c, b, radius=0.3, fill=True, padding=None ):
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        img = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        img = np.zeros((int(x),int(y),1), np.uint8)
                                   
    if padding is not None:   
        cx = int(c[0]/r) + int(padding/r)
        cy = int(c[1]/r) + int(padding/r)
    else:       
        cx = int(c[0]/r)
        cy = int(c[1]/r)
        
    radius = int(radius / r)
    
    if fill == True:
        thickness = -1 
    else:
        thickness = 1
        
    cv2.circle(img=img,
               center=(cx,cy), 
               color=(255),
               radius=radius,
               thickness = thickness )
        
    if padding is not None:
        return cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:
        return img
    
def draw_circle( c, b, radius=0.3, fill=True, padding=None):
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        img = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        img = np.zeros((int(x),int(y),1), np.uint8)
                                   
    if padding is not None:   
        cx = int(c[0]/r) + int(padding/r)
        cy = int(c[1]/r) + int(padding/r)
    else:       
        cx = int(c[0]/r)
        cy = int(c[1]/r)

    radius = int(radius / r)
    
    if fill == True:
        thickness = -1 
    else:
        thickness = 1
        
    cv2.circle(img=img,
               center=(cx,cy), 
               color=(255),
               radius=radius,
               thickness = thickness )
        
    if padding is not None:
        return cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:
        return img
    
    
def draw_line_between_points( p1, p2, b, thickness=1, padding=None ):
    # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        img = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        img = np.zeros((int(x),int(y),1), np.uint8)
                                   
    if padding is not None:   
        p1x = int(p1[0]/r) + int(padding/r)
        p1y = int(p1[1]/r) + int(padding/r)
        p2x = int(p2[0]/r) + int(padding/r)
        p2y = int(p2[1]/r) + int(padding/r)
        
    else:       
        p1x = int(p1[0]/r)
        p1y = int(p1[1]/r)
        p2x = int(p2[0]/r)
        p2y = int(p2[1]/r)
                
    cv2.line(img=img,
               pt1=(p1x,p1y), 
               pt2=(p2x,p2y), 
               color=(255),
               thickness = 1 )
        
    if padding is not None:
        return cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0))
    else:
        return img

    
def calculate_resolution(b, padding=None):
    x = int(b.get_width() / r)
    y = int(b.get_height() / r)
    
    if padding is not None:
        x += 2*int(padding/r)
        y += 2*int(padding/r) 
        
    return (x,y)

### Environment pcb_component_w_vec_distance_v2 ###

def coords_grid_to_node(pos, resolution, padding=None):
        if padding is not None:
            return ((pos[0]- (padding/resolution)) * resolution, 
                    (pos[1] - (padding/resolution)) * resolution)
        else:
            return (pos[0] * resolution, 
                    pos[1] * resolution)                    

def coords_node_to_grid(pos, resolution, padding=None):
        if padding is not None:
            return (np.int0(pos[0] / resolution) + int(padding / resolution),  
                    np.int0(pos[1] / resolution) + int(padding / resolution))
        else:
            return (np.int0(pos[0] / resolution), 
                    np.int0(pos[1] / resolution))   

def setup_empty_grid(b, r, padding=None):
        # Setup grid
    x = b.get_width() / r;
    y = b.get_height() / r;
    
    if padding is not None:
        grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid = np.zeros((int(x),int(y),1), np.uint8)           

    return grid     

def setup_empty_grid(bx, by, r, padding=None):
        # Setup grid
    x = bx / r;
    y = by / r;
    
    if padding is not None:
        grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid = np.zeros((int(x),int(y),1), np.uint8)           

    return grid    

def get_los_and_ol(node, board, radius, grid_comps, padding, los_type=0):
    # type 0 - traditional case
    # type 1 - remove current node from the radius.
    # type 3 - cropped grid showing overlapping section
    # type 4 - cropped grid showing overlapping section and current node.
    
    angle_offset = node.get_orientation()
    r = pcbDraw_resolution()
    x = board.get_width() / r
    y = board.get_height() / r
    pos = node.get_pos()
    
    if padding is not None:   
        cx = int(pos[0]/r) + int(padding/r)
        cy = int(pos[1]/r) + int(padding/r)
    else:       
        cx = int(pos[0]/r)
        cy = int(pos[1]/r)
        
    radius = int(radius / r)
    
    if los_type == 0 or los_type == 1:
        
        if padding is not None:
            los_segments_mask = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            los_segments = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            overlap_segments_mask = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            overlap_segments = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)

        else:
            los_segments_mask = np.zeros((8,int(x),int(y)), np.uint8)
            los_segments = np.zeros((8,int(x),int(y)), np.uint8)
            overlap_segments_mask = np.zeros((8,int(x),int(y)), np.uint8)
            overlap_segments = np.zeros((8,int(x),int(y)), np.uint8)

        segment_mask_pixels = np.zeros(8)
        segment_pixels = np.zeros(8)
        overlap_mask_pixels = np.zeros(8)
        overlap_pixels = np.zeros(8)
        start = -22.5 - angle_offset
        stop = 22.5 - angle_offset
        for i in range(8):
            cv2.ellipse(los_segments_mask[i], (cx,cy), (radius,radius), 0, start, stop, (64), -1) 
            overlap_segments_mask[i] = cv2.bitwise_and(src1=los_segments_mask[i], src2=grid_comps[1])
            overlap_mask_pixels[i] = np.sum(overlap_segments_mask[i])
            if los_type == 1:
                los_segments_mask[i] -= overlap_segments_mask[i]
           
            segment_mask_pixels[i] = np.sum(los_segments_mask[i])
            
            los_segments[i] = cv2.bitwise_and(src1=los_segments_mask[i], src2=grid_comps[0])
            overlap_segments[i] = cv2.bitwise_and(src1=overlap_segments_mask[i], src2=grid_comps[0])
            segment_pixels[i] = np.sum(los_segments[i])
            overlap_pixels[i] = np.sum(overlap_segments[i])

            start -= 45 
            stop -= 45
            
        
        return segment_pixels/segment_mask_pixels, overlap_pixels/overlap_mask_pixels, los_segments_mask, overlap_segments_mask  
    
    if los_type == 2:
        return
    
    if los_type == 3 or los_type == 4:
        grid = setup_empty_grid(bx=board.get_width(), by=board.get_height(), r=pcbDraw_resolution(), padding=padding)

        cv2.circle(img=grid,
                    center=(cx,cy), 
                    color=(64),
                    radius=radius,
                    thickness = -1 )
        
        grid = grid.reshape(grid.shape[0], grid.shape[1])
        grid = cv2.bitwise_and(src1=grid, src2=grid_comps[0])
        if los_type == 3:
            return grid[int(cy-radius/2-1):int(cy+radius/2+1) ,int(cx-radius/2-1):int(cx+radius/2+1)]
        else:
            grid += grid_comps[1]
            return grid[int(cy-radius/2-1):int(cy+radius/2+1) ,int(cx-radius/2-1):int(cx+radius/2+1)]

def get_los_and_ol_multi_agent(node, board, radius, grid_comps, padding, los_type=0):
    # type 0 - traditional case
    # type 1 - remove current node from the radius.
    # type 3 - cropped grid showing overlapping section
    # type 4 - cropped grid showing overlapping section and current node.
    
    angle_offset = node.get_orientation()
    r = pcbDraw_resolution()
    x = board.get_width() / r
    y = board.get_height() / r
    pos = node.get_pos()
    
    if padding is not None:   
        cx = int(pos[0]/r) + int(padding/r)
        cy = int(pos[1]/r) + int(padding/r)
    else:       
        cx = int(pos[0]/r)
        cy = int(pos[1]/r)
        
    radius = int(radius / r)
    
    if los_type == 0 or los_type == 1:
        
        if padding is not None:
            los_segments_mask = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            los_segments = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            los_segments_debug = np.zeros((1,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)

            overlap_segments_mask = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            overlap_segments = np.zeros((8,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)
            overlap_segments_debug = np.zeros((1,int(x)+2*int(padding/r),int(y)+2*int(padding/r)), np.uint8)


        else:
            los_segments_mask = np.zeros((8,int(x),int(y)), np.uint8)
            los_segments = np.zeros((8,int(x),int(y)), np.uint8)
            overlap_segments_mask = np.zeros((8,int(x),int(y)), np.uint8)
            overlap_segments = np.zeros((8,int(x),int(y)), np.uint8)

        segment_mask_pixels = np.zeros(8)
        segment_pixels = np.zeros(8)
        overlap_mask_pixels = np.zeros(8)
        overlap_pixels = np.zeros(8)
        start = -22.5 - angle_offset
        stop = 22.5 - angle_offset
        for i in range(8):
            cv2.ellipse(los_segments_mask[i], (cx,cy), (radius,radius), 0, start, stop, (64), -1) 
            overlap_segments_mask[i] = cv2.bitwise_and(src1=los_segments_mask[i], src2=grid_comps[1])
            overlap_mask_pixels[i] = np.sum(overlap_segments_mask[i])
            if los_type == 1:
                los_segments_mask[i] -= overlap_segments_mask[i]
           
            segment_mask_pixels[i] = np.sum(los_segments_mask[i])
            
            for j in range(2, len(grid_comps),1):
                los_segments[i] = cv2.bitwise_or(src1=cv2.bitwise_and(src1=los_segments_mask[i], 
                                                                      src2=grid_comps[j]),
                                                 src2=los_segments[i])
                
                           
                overlap_segments[i] = cv2.bitwise_or(src1=cv2.bitwise_and(src1=overlap_segments_mask[i],
                                                                          src2=grid_comps[j]),
                                                     src2=overlap_segments[i])
                
            los_segments[i] = cv2.bitwise_or(src1=cv2.bitwise_and(src1=los_segments_mask[i], 
                                                                  src2=grid_comps[0]),
                                            src2=los_segments[i])
                                            
            overlap_segments[i] = cv2.bitwise_or(src1=cv2.bitwise_and(src1=overlap_segments_mask[i],
                                                                      src2=grid_comps[0]),
                                                 src2=overlap_segments[i])                                            
                
            # los_segments_debug += los_segments[i]
            # overlap_segments_debug += overlap_segments[i]
            segment_pixels[i] = np.sum(los_segments[i])
            overlap_pixels[i] = np.sum(overlap_segments[i])

            start -= 45 
            stop -= 45
         
        # print(np.round(segment_pixels/segment_mask_pixels,2))
        # print(np.round(overlap_pixels/overlap_mask_pixels,2))
        # print()   
         
        # if node.get_id() == 0:
        #     cv2.imshow("get_los_and_ol_multi_agent_0", cv2.hconcat([grid_comps[0]+grid_comps[1]+grid_comps[2]+grid_comps[3], los_segments_debug[0],overlap_segments_debug[0]]))
        #     cv2.waitKey()
        # elif node.get_id() == 1:
        #     cv2.imshow("get_los_and_ol_multi_agent_1", cv2.hconcat([grid_comps[0]+grid_comps[1]+grid_comps[2]+grid_comps[3], los_segments_debug[0],overlap_segments_debug[0]]))
        #     cv2.waitKey()

        
        return segment_pixels/segment_mask_pixels, overlap_pixels/overlap_mask_pixels, los_segments_mask, overlap_segments_mask  
    
    if los_type == 2:
        return
    
    if los_type == 3 or los_type == 4:
        grid = setup_empty_grid(bx=board.get_width(), by=board.get_height(), r=pcbDraw_resolution(), padding=padding)

        cv2.circle(img=grid,
                    center=(cx,cy), 
                    color=(64),
                    radius=radius,
                    thickness = -1 )
        
        grid = grid.reshape(grid.shape[0], grid.shape[1])
        grid = cv2.bitwise_and(src1=grid, src2=grid_comps[0])
        if los_type == 3:
            return grid[int(cy-radius/2-1):int(cy+radius/2+1) ,int(cx-radius/2-1):int(cx+radius/2+1)]
        else:
            grid += grid_comps[1]
            return grid[int(cy-radius/2-1):int(cy+radius/2+1) ,int(cx-radius/2-1):int(cx+radius/2+1)]
