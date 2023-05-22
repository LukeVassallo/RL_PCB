#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:38:05 2022

@author: luke
"""

import numpy as np
import cv2
from graph import graph
from graph import board
from graph import node
from graph import edge
from graph_utils import kicad_rotate
import sys

r = 0.1    # resolution in mm

def draw_board_from_board_and_graph(b,
                                    g,
                                    draw_placed=True,
                                    draw_unplaced=True,
                                    padding=None,
                                    line_thickness=-1):
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
        Contains two grid with grayscale drawings. In position 0 there is a
        drawing containing the board with the placed components. Position 1
        contains the unplaced component.
    """
    nv = g.get_nodes()

    # Setup grid
    x = b.get_width() / r
    y = b.get_height() / r

    if padding is not None:
        grid_comps = np.zeros(
            (2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)

    for n in nv:
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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        if n.get_isPlaced() == 1:
            cv2.drawContours(grid_comps[0],[box],0,(64),line_thickness)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),line_thickness)

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        return [cv2.copyMakeBorder(grid_comps[0],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(64)) + border,
              cv2.copyMakeBorder(grid_comps[1],
                                 0, 0, 0, 0,
                                 cv2.BORDER_CONSTANT,
                                 value=(0))]
    else:
        return grid_comps

def draw_board_from_board_and_graph_with_debug(b,
                                               g,
                                               draw_placed=True,
                                               draw_unplaced=True,
                                               padding=None,
                                               line_thickness=-1):
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
        Contains two grid with grayscale drawings. In position 0 there is a
        drawing containing the board with the placed components. Position 1
        contains the unplaced component.
    """
    nv = g.get_nodes()

    # Setup grid
    x = b.get_width() / r
    y = b.get_height() / r

    if padding is not None:
        grid_comps = np.zeros(
            (3,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        grid_comps = np.zeros((3,int(x),int(y),1), np.uint8)

    for n in nv:
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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        if n.get_isPlaced() == 1:
            cv2.drawContours(grid_comps[0],[box],0,(64),line_thickness)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),line_thickness)

        if padding is not None:
            tmp = draw_node_name(n,
                                 b.get_width(),
                                 b.get_height(),
                                 padding=padding)
            tmp = np.reshape(tmp,(tmp.shape[0],tmp.shape[1],1))
            grid_comps[2] = np.maximum(tmp, grid_comps[2])

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        return [cv2.copyMakeBorder(grid_comps[0],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(64)) + border,
                cv2.copyMakeBorder(grid_comps[1],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(0)),
                cv2.copyMakeBorder(grid_comps[2],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(0))
                                   ]
    else:
        return grid_comps

def draw_board_from_board_and_graph_multi_agent(b, g,
                                                node_id,
                                                draw_placed=True,
                                                draw_unplaced=True,
                                                padding=None,
                                                line_thickness=-1):
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
        Contains two grid with grayscale drawings. In position 0 there is a
        drawing containing the board with the placed components. Position 1
        contains the unplaced component.
    """
    nv = g.get_nodes()

    # Setup grid
    x = b.get_width() / r
    y = b.get_height() / r

    if padding is not None:
        grid_comps = np.zeros(
            (len(nv)+1,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        grid_comps = np.zeros((len(nv)+1,int(x),int(y),1), np.uint8)
    idx = 2
    for n in nv:
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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        if n.get_id() == node_id:
            cv2.drawContours(grid_comps[1],[box],0,(64),line_thickness)
        else:
            cv2.drawContours(grid_comps[idx],[box],0,(64),line_thickness)
            idx +=1

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        stack = []

        stack.append(cv2.copyMakeBorder(grid_comps[0],
                                        0, 0, 0, 0,
                                        cv2.BORDER_CONSTANT,
                                        value=(64)) + border)
        for i in range(1, idx, 1):
            stack.append(cv2.copyMakeBorder(grid_comps[i],
                                            0, 0, 0, 0,
                                            cv2.BORDER_CONSTANT,
                                            value=(0)))
        return stack
    else:
        print("draw_board_from_board_and_graph_multi_agent requires padding.")
        sys.exit()

def draw_comps_from_nodes_and_edges(n, nn, e, b, padding=None):
    # Setup grid
    x = b.get_width() / r
    y = b.get_height() / r

    if padding is not None:
        grid_comps = np.zeros(
            (2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1), np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)

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
    # convert the center, size and orientation to rectange points
    box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
    box = np.int0(box)  # ensure that box point are integers
    if n.get_isPlaced() == 1: # should always return False!
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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        if v.get_isPlaced() == 1:  # should always return True!
            cv2.drawContours(grid_comps[0],[box],0,(64),-1)
        else:
            cv2.drawContours(grid_comps[1],[box],0,(64),-1)

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        return [cv2.copyMakeBorder(grid_comps[0],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(64)) + border,
                cv2.copyMakeBorder(grid_comps[1],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(0))]
    else:
        return grid_comps

# The following is the difference between this and the prior method
# neighbor nodes are always treated as 'locked'
# current node is always treated as 'unlocked'
def draw_board_from_nodes_and_edges_multi_agent(n,
                                                nn,
                                                e,
                                                bx,
                                                by,
                                                padding=None):
    # Setup grid
    x = bx / r
    y = by / r

    if padding is not None:
        grid_comps = np.zeros(
            (len(nn)+2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
             np.uint8)
    else:
        grid_comps = np.zeros((len(nn)+2,int(x),int(y),1), np.uint8)

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
    # convert the center, size and orientation to rectange points
    box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        cv2.drawContours(grid_comps[idx],[box],0,(64),-1)
        idx += 1

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        tmp =[]
        for i in range(len(nn)+2):
            if i == 0:
                tmp.append(cv2.copyMakeBorder(
                    grid_comps[0],
                    0, 0, 0, 0,
                    cv2.BORDER_CONSTANT,
                    value=(64)) + border)
            else:
                tmp.append(cv2.copyMakeBorder(
                    grid_comps[i],
                    0, 0, 0, 0,
                    cv2.BORDER_CONSTANT,
                    value=(0)))
        return tmp
    else:
        return grid_comps

# idx = 0 ( grid border )
# idx = 1 ( current node  )
# idx = 2 ... ( neighbors ... )
def draw_board_from_graph_multi_agent(g, node_id, bx, by, padding=None):
    # Setup grid
    x = bx / r
    y = by / r

    all_nodes = g.get_nodes()

    if padding is not None:
        grid_comps = np.zeros(
            (len(all_nodes)+1,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        grid_comps = np.zeros((len(all_nodes)+1,int(x),int(y),1), np.uint8)

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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        if current_node_id == node_id:
            cv2.drawContours(grid_comps[1],[box],0,(64),-1)
        else:
            cv2.drawContours(grid_comps[idx],[box],0,(64),-1)
            idx += 1

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        tmp =[]
        for i in range(len(all_nodes)+1):
            if i == 0:
                tmp.append(cv2.copyMakeBorder(grid_comps[0],
                                              0, 0, 0, 0,
                                              cv2.BORDER_CONSTANT,
                                              value=(64)) + border)
            else:
                tmp.append(cv2.copyMakeBorder(grid_comps[i],
                                              0, 0, 0, 0,
                                              cv2.BORDER_CONSTANT,
                                              value=(0)))

        return tmp
    else:
        return grid_comps

# only comp_grids[0] is used.
def draw_board_from_nodes_multi_agent(n, bx, by, padding=None):
    # Setup grid
    x = bx / r
    y = by / r

    if padding is not None:
        grid_comps = np.zeros(
            (2,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        grid_comps = np.zeros((2,int(x),int(y),1), np.uint8)

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
        # convert the center, size and orientation to rectange points
        box = cv2.boxPoints(((xc,yc), (sz_x,sz_y), -orientation))
        box = np.int0(box)  # ensure that box point are integers
        cv2.drawContours(grid_comps[0],[box],0,(64),-1)

    if padding is not None:
        border = np.zeros((int(x),int(y),1), np.uint8)
        border = cv2.copyMakeBorder(border,
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    int(padding/r),
                                    cv2.BORDER_CONSTANT,
                                    value=(64))

        return [cv2.copyMakeBorder(grid_comps[0],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(64)) + border,
              cv2.copyMakeBorder(grid_comps[1],
                                 0, 0, 0, 0,
                                 cv2.BORDER_CONSTANT,
                                 value=(0))]
    else:
        return grid_comps

def draw_ratsnest_with_board(current_node,
                             neighbor_nodes,
                             e,
                             b,
                             line_thickness=1,
                             padding=None,
                             ignore_power=False):
    # Setup grid
    bx = b.get_width()
    by = b.get_height()

    return draw_ratsnest(current_node,
                         neighbor_nodes,
                         e,
                         bx,
                         by,
                         line_thickness=line_thickness,
                         padding=padding,
                         ignore_power=ignore_power)

def draw_ratsnest(current_node,
                  neighbor_nodes,
                  e,
                  bx,
                  by,
                  line_thickness=1,
                  padding=None,
                  ignore_power=False):
    x = bx / r
    y = by / r

    if padding is not None:
        ratsnest = np.zeros(
            (int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        ratsnest = np.zeros((int(x),int(y),1), np.uint8)

    # draw current node
    current_node_id = current_node.get_id()
    current_node_pos = current_node.get_pos()
    current_node_orientation = current_node.get_orientation()

    src = []
    dst = []
    for ee in e:
        if (ignore_power is True) and (ee.get_power_rail() > 0):
            continue
        # Get list of pos, size for all
        for i in range(2):
            if ee.get_instance_id(i) == current_node_id:
                pad_pos = ee.get_pos(i)
                # rotate pad positions so that they match the component's
                # orientation
                rotated_pad_pos = kicad_rotate(float(pad_pos[0]),
                                               float(pad_pos[1]),
                                               current_node_orientation)
                src.append([current_node_pos[0] + rotated_pad_pos[0],
                            current_node_pos[1] + rotated_pad_pos[1]
                            ])
            else:
                for n in neighbor_nodes:
                    if ee.get_instance_id(i) == n.get_id():
                        neighbor_node_pos = n.get_pos()
                        pad_pos = ee.get_pos(i)
                        # rotate pad positions so that they match the
                        # component's orientation
                        rotated_pad_pos = kicad_rotate(float(pad_pos[0]),
                                                       float(pad_pos[1]),
                                                       n.get_orientation())

                        dst.append([neighbor_node_pos[0] + rotated_pad_pos[0],
                                    neighbor_node_pos[1] + rotated_pad_pos[1]
                                    ])
                        break

    for i in range(min(len(src),len(dst))):
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

        # image, pt1 (x,y), pt2, color (BGR), thickness
        cv2.line(ratsnest,
                 (sx,sy),
                 (dx,dy),
                 (255),
                 line_thickness)

    if padding is not None:
        return cv2.copyMakeBorder(ratsnest,
                                  0, 0, 0, 0,
                                  cv2.BORDER_CONSTANT,
                                  value=(0))
    else:
        return ratsnest

def draw_los(pos_x,
             pos_y,
             radius,
             angle_offset,
             bx,
             by,
             padding=None):
    """
    Parameters
    ----------
    pos_x : TYPE
        floating point node x-co-ordinate. This show not be formatted into
        pixels
    pos_y : TYPE
        floating point node y-co-ordinate. This show not be formatted into
        pixels
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
    x = bx / r
    y = by / r
    if padding is not None:
        los_segments = np.zeros(
            (8,int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
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
    start = -22.5 - angle_offset
    stop = 22.5 - angle_offset
    for i in range(8):
        cv2.ellipse(los_segments[i],
                    (scaled_x,scaled_y),
                    (scaled_radius,scaled_radius),
                    0,
                    start,
                    stop,
                    (16),
                    -1)
        segment_pixels[i] = np.sum(los_segments[i]) / 16
        start -= 45
        stop -= 45

    # this is dumb, but needed so arrays are matching.
    # copyMakeBorder reshapes the output image.
    if padding is not None:
        for i in range(8):
            padded_los_segments.append(
                cv2.copyMakeBorder(los_segments[i],
                                   0, 0, 0, 0,
                                   cv2.BORDER_CONSTANT,
                                   value=(0))
                                   )

        return np.array(padded_los_segments), segment_pixels
    else:
        return los_segments, segment_pixels

def draw_node_name(n,
                   bx,
                   by,
                   padding=None,
                   loc="top_right",
                   designator_only=False):
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

    x = bx / r
    y = by / r
    if padding is not None:
        grid = np.zeros(
            (int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
            np.uint8)
    else:
        grid = np.zeros((int(x),int(y),1), np.uint8)

    if loc == "top_left":
        if padding is None:
            # - 1 since text moves from left to right
            text_origin = (
                int( (n.get_pos()[0] - n.get_size()[0]/2- 1)/r ),
                int( (n.get_pos()[1] - n.get_size()[1]/2 - 0.25 )/r ))
        else:
            # - 1 since text moves from left to right
            text_origin = (
                int((n.get_pos()[0] - n.get_size()[0]/2 - 1)/r) + int(padding/r),
                int((n.get_pos()[1] - n.get_size()[1]/2 - 0.25)/r) + int(padding/r))
    else:# loc == "top_right":
        # place orientation on the top right
        if padding is None:
            text_origin = (int((n.get_pos()[0] + n.get_size()[0]/2)/r),
                        int((n.get_pos()[1] - n.get_size()[1]/2)/r))
        else:
            text_origin = (
                int((n.get_pos()[0] + n.get_size()[0]/2)/r) + int(padding/r),
                int((n.get_pos()[1] - n.get_size()[1]/2)/r) + int(padding/r))

    if designator_only is True:
        cv2.putText(img=grid,
            text=f"{n.get_name()}",
            org=text_origin,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(127))
    else:
        cv2.putText(img=grid,
                    text=f"{n.get_id()} ({n.get_name()})",
                    org=text_origin,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=127)

    # this is dumb, but needed so arrays are matching.
    # copyMakeBorder reshapes the output image.
    if padding is not None:
        return cv2.copyMakeBorder(grid,
                                  0, 0, 0, 0,
                                  cv2.BORDER_CONSTANT,
                                  value=0)
    else:
        return grid

def pcbDraw_resolution():
    return r

def set_pcbDraw_resolution(resolution):
    global r
    r = resolution

def setup_empty_grid(bx, by, resolution, padding=None):
    # Setup grid
    x = bx / resolution
    y = by / resolution

    if padding is not None:
        grid = np.zeros((int(x)+2*int(padding/r),int(y)+2*int(padding/r),1),
                        np.uint8)
    else:
        grid = np.zeros((int(x),int(y),1), np.uint8)

    return grid

def get_los_and_ol_multi_agent(node,
                               board,
                               radius,
                               grid_comps,
                               padding,
                               los_type=0):
    # type 0 - traditional case
    # type 1 - remove current node from the radius.
    # type 3 - cropped grid showing overlapping section
    # type 4 - cropped grid showing overlapping section and current node.

    angle_offset = node.get_orientation()
    res = pcbDraw_resolution()
    x = board.get_width() / res
    y = board.get_height() / res
    pos = node.get_pos()

    if padding is not None:
        cx = int(pos[0]/res) + int(padding/res)
        cy = int(pos[1]/res) + int(padding/res)
    else:
        cx = int(pos[0]/res)
        cy = int(pos[1]/res)

    radius = int(radius / res)

    if los_type in (0, 1):
        if padding is not None:
            los_segments_mask = np.zeros(
                (8,int(x)+2*int(padding/res),int(y)+2*int(padding/res)),
                np.uint8)
            los_segments = np.zeros(
                (8,int(x)+2*int(padding/res),int(y)+2*int(padding/res)),
                np.uint8)
            overlap_segments_mask = np.zeros(
                (8,int(x)+2*int(padding/res),int(y)+2*int(padding/res)),
                np.uint8)
            overlap_segments = np.zeros(
                (8,int(x)+2*int(padding/res),int(y)+2*int(padding/res)),
                np.uint8)
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
            cv2.ellipse(los_segments_mask[i],
                        (cx,cy),
                        (radius,radius),
                        0,
                        start,
                        stop,
                        (64),
                        -1)
            overlap_segments_mask[i] = cv2.bitwise_and(
                src1=los_segments_mask[i],
                src2=grid_comps[1])

            overlap_mask_pixels[i] = np.sum(overlap_segments_mask[i])
            if los_type == 1:
                los_segments_mask[i] -= overlap_segments_mask[i]

            segment_mask_pixels[i] = np.sum(los_segments_mask[i])

            for j in range(2, len(grid_comps),1):
                los_segments[i] = cv2.bitwise_or(
                    src1=cv2.bitwise_and(src1=los_segments_mask[i],
                                         src2=grid_comps[j]),
                    src2=los_segments[i])

                overlap_segments[i] = cv2.bitwise_or(
                    src1=cv2.bitwise_and(src1=overlap_segments_mask[i],
                                         src2=grid_comps[j]),
                    src2=overlap_segments[i])

            los_segments[i] = cv2.bitwise_or(
                src1=cv2.bitwise_and(src1=los_segments_mask[i],
                                     src2=grid_comps[0]),
                src2=los_segments[i])
            overlap_segments[i] = cv2.bitwise_or(
                src1=cv2.bitwise_and(src1=overlap_segments_mask[i],
                                     src2=grid_comps[0]),
                src2=overlap_segments[i])

            segment_pixels[i] = np.sum(los_segments[i])
            overlap_pixels[i] = np.sum(overlap_segments[i])

            start -= 45
            stop -= 45

        return segment_pixels/segment_mask_pixels, overlap_pixels/overlap_mask_pixels, los_segments_mask, overlap_segments_mask

    if los_type == 2:
        return

    if los_type in (3, 4):
        grid = setup_empty_grid(bx=board.get_width(),
                                by=board.get_height(),
                                resolution=pcbDraw_resolution(),
                                padding=padding)

        cv2.circle(img=grid,
                    center=(cx,cy),
                    color=(64),
                    radius=radius,
                    thickness = -1 )

        grid = grid.reshape(grid.shape[0], grid.shape[1])
        grid = cv2.bitwise_and(src1=grid, src2=grid_comps[0])
        if los_type == 3:
            return grid[int(cy-radius/2-1):int(cy+radius/2+1),
                        int(cx-radius/2-1):int(cx+radius/2+1)]
        else:
            grid += grid_comps[1]
            return grid[int(cy-radius/2-1):int(cy+radius/2+1),
                        int(cx-radius/2-1):int(cx+radius/2+1)]
