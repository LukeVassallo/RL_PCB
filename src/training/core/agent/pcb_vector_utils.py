#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:33:31 2022

@author: luke
"""
import numpy as np
from graph_utils import kicad_rotate
from pcbDraw import draw_los, draw_comps_from_nodes_and_edges, pcbDraw_resolution

def polar_to_rectangular(r, theta):
    return r * np.exp( 1j * theta )

def rectangular_to_polar(z):
    r = np.abs(z)
    theta = np.angle(z)
    return (r, theta)

def compute_pad_referenced_distance_vectors_v2(n, nn, e, ignore_power=False):
    """
    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    nn : TYPE
        DESCRIPTION.
    e : TYPE
        DESCRIPTION.

    Returns
    -------
    dom : TYPE
        DESCRIPTION.
    resultant_vecs : TYPE
        DESCRIPTION.
        List of list. The latter containing [ net_id, current_node_id,
        target_node_id, current_pad_id, target_pad_id ] (r, theta)
    all_vecs : TYPE
        DESCRIPTION.
        List of lists. Each list contains a list of vectors correspond to a
        specific current_node - target_node pair.
        The latter containing [ net_id, current_node_id, target_node_id,
        current_pad_id, target_pad_id ] (current_pad_x, current_pad_y,
        target_pad_x, target_pad_y, r, theta)


    The function does the following:
        #1 Compute connection vectors from current component pad to target
        neighbor pad. In case multiple pads are involved in a given net, use
        the one with the shortest distance.
        #2 Reduce the vectors between components by summing them up (i.e.
        obtain the resultant vector between the current component and each
        of its neighbors)
        #3 Obtain the direction of movement by summing up all the resultant
        vectors.

        Notes added after the implementation but no changes to the code were
        carried out. The magnitude of the resultant vector is divded by the
        number of vectors in the net.
    """
    current_node_id = n.get_id()
    current_node_pos = n.get_pos()
    net_ids = []

    # 1 Make a list containing pairs of data points for every net
    for ee in e:
        if (ignore_power is True) and (ee.get_power_rail() > 0):
            continue
        if ee.get_net_id() not in net_ids:
            net_ids.append(ee.get_net_id())

    pts = []
    for net_id in net_ids:
        for ee in e:
            if ee.get_net_id() == net_id:
                for i in range(2):
                    if ee.get_instance_id(i) == current_node_id:
                        current_pad_pos = ee.get_pos(i)
                        # rotate pad positions so that they match the
                        # component's orientation
                        rotated_current_pad_pos = kicad_rotate(
                            float(current_pad_pos[0]),
                            float(current_pad_pos[1]),
                            n.get_orientation())

                        neighbor_pad_pos = ee.get_pos(1-i)
                        for v in nn:
                            if v.get_id() == ee.get_instance_id(1-i):
                                # rotate pad positions so that they match the
                                # component's orientation
                                rotated_neighbor_pad_pos = kicad_rotate(
                                    float(neighbor_pad_pos[0]),
                                    float(neighbor_pad_pos[1]),
                                    v.get_orientation())

                                neighbor_node_pos = v.get_pos()

                                header = [net_id,
                                            current_node_id,
                                            v.get_id(),
                                            ee.get_pad_id(i),      # current node pad id
                                            ee.get_pad_id(1-i)]    # target node pad id
                                break

                        sx = current_node_pos[0] + rotated_current_pad_pos[0]
                        sy = current_node_pos[1] + rotated_current_pad_pos[1]
                        dx = neighbor_node_pos[0] + rotated_neighbor_pad_pos[0]
                        dy = neighbor_node_pos[1] + rotated_neighbor_pad_pos[1]

                        delta_y = (sy-dy)
                        delta_x = (dx-sx)

                        euclidean_dist = np.sqrt(
                            np.square(delta_x) + np.square(delta_y))
                        angle = np.arctan(delta_y/delta_x)
                        if delta_x < 0: angle += np.pi

                        # 2 Remove duplicates by taking the shorter ones
                        found = False
                        for j in range(len(pts)):
                            if pts[j][0:4] == header[0:4]:
                                found = True
                                if pts[j][-2] > euclidean_dist:
                                    pts[j] = header + [
                                                sx,
                                                sy,
                                                dx,
                                                dy,
                                                euclidean_dist,
                                                angle
                                                ]
                                    break

                        if not found:
                            pts.append(
                                header + [sx, sy, dx, dy, euclidean_dist, angle])

    # 3 Compute vectors
    # p[2:3] -> current_node_id, current_pad_id
    vec_sc = []
    for p in pts:
        if [p[1], p[3]] not in vec_sc:
            vec_sc.append([p[1], p[3]])

    all_vecs = []
    for i in vec_sc:
        tmp = []
        for p in pts:
            if [p[1], p[3]] == i:
                tmp.append(p)
        all_vecs.append(tmp)
    resultant_vecs = []
    # The simplest way to add two polar vectors is by converting them to
    # rectangular form, summing them up and converting them back to polar.
    for vecs in all_vecs:
        v_pts = []
        for v in vecs:
            # Divide the magnitude of a vector the sum of vectors in the list.
            v_pts.append(polar_to_rectangular(v[-2]/len(pts),v[-1]))

        z = rectangular_to_polar(np.sum(v_pts))
        resultant_vecs.append([v[0:5], z])

    v_pts = []
    for v in resultant_vecs:
        v_pts.append(polar_to_rectangular(v[-1][0], v[-1][1]))

    dom = rectangular_to_polar(np.sum(v_pts))
    return dom, resultant_vecs, all_vecs

def sort_resultant_vectors( resultant_vecs ):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    resultant_vecs.sort(key = lambda x: x[0][3])
    return resultant_vecs

def compute_vector_to_group_midpoint(n, nn):
    all_pos_x = []
    all_pos_y = []

    current_node_pos = n.get_pos()
    all_pos_x.append(current_node_pos[0])
    all_pos_y.append(current_node_pos[1])

    for v in nn:
        pos = v.get_pos()
        all_pos_x.append(pos[0])
        all_pos_y.append(pos[1])

    cx = np.sum(np.array(all_pos_x))/len(all_pos_x)
    cy = np.sum(np.array(all_pos_y))/len(all_pos_y)

    delta_y = (current_node_pos[1]-cy)
    delta_x = (cx-current_node_pos[0])

    euclidean_dist = np.sqrt(np.square(delta_x) + np.square(delta_y))
    angle = np.arctan(delta_y/delta_x)
    if delta_x < 0: angle += np.pi

    return tuple([cx,cy]), euclidean_dist, angle

# computes the sum of shortest distances between the current node and
# it's neighbors
def compute_sum_of_euclidean_distances(n, nn, eoi):
    current_node_id = n.get_id()
    current_node_pos = n.get_pos()
    current_node_orientation = n.get_orientation()
    all_lengths = []
    for v in nn:
        neighbor_node_id = v.get_id()
        lengths = []
        for e in eoi:
            if (e.get_instance_id(0) == current_node_id or e.get_instance_id(0) == neighbor_node_id) and (e.get_instance_id(1) == current_node_id or e.get_instance_id(1) == neighbor_node_id):
                for i in range(2):
                    if e.get_instance_id(i) == current_node_id:
                        current_pad_pos = e.get_pos(i)
                        rotated_current_pad_pos = kicad_rotate(
                            float(current_pad_pos[0]),
                            float(current_pad_pos[1]),
                            current_node_orientation)

                        neighbor_pad_pos = e.get_pos(1-i)
                        # rotate pad positions so that they match the
                        # component's orientation
                        rotated_neighbor_pad_pos = kicad_rotate(
                            float(neighbor_pad_pos[0]),
                            float(neighbor_pad_pos[1]),
                            v.get_orientation())
                        neighbor_node_pos = v.get_pos()

                        p1 = [current_node_pos[0] + rotated_current_pad_pos[0], current_node_pos[1] + rotated_current_pad_pos[1]]
                        p2 = [neighbor_node_pos[0] + rotated_neighbor_pad_pos[0], neighbor_node_pos[1] + rotated_neighbor_pad_pos[1]]

                        lengths.append(np.sqrt(np.square(p1[0]-p1[1])+np.square(p2[0]-p2[1])))

        all_lengths.append(np.min(np.array(lengths)))
        print(f" Length array between current_node {current_node_id},{n.get_name()} and neighbor node  {v.get_id()},{v.get_name()} is : {lengths}")
        print(f"The smallest value being {all_lengths[-1]}")

    print(np.sum(all_lengths))

# computes the sum of shortest distances between the current node's pads and
# it's neighbors. This is achieved by first computing the euclidean distance
# of all edges. These represent a pad to pad connection. For each pad of the
# current node the shortest pad-pad distance to any of its neighbors is noted.
# The sum is returned.
def compute_sum_of_euclidean_distances_between_pads(n,
                                                    nn,
                                                    eoi,
                                                    ignore_power=False):
    current_node_id = n.get_id()
    current_node_pos = n.get_pos()
    current_node_orientation = n.get_orientation()
    current_node_pins = n.get_pin_count()

    all_lengths = []
    for i in range(current_node_pins):
        lengths = []
        for e in eoi:
            if ignore_power is True and e.get_power_rail() > 0:
                continue

            for j in range(2):
                if (e.get_instance_id(j) == current_node_id) and (e.get_pad_id(j) == i):
                    for k in range(2):
                        if (e.get_instance_id(k) == current_node_id) and (e.get_pad_id(k) == i):
                            current_pad_pos = e.get_pos(k)
                            rotated_current_pad_pos = kicad_rotate(
                                float(current_pad_pos[0]),
                                float(current_pad_pos[1]),
                                current_node_orientation)

                            for v in nn:
                                if v.get_id() == e.get_instance_id(1-k):
                                    neighbor_pad_pos = e.get_pos(1-k)
                                    # rotate pad positions so that they match
                                    # the component's orientation
                                    rotated_neighbor_pad_pos = kicad_rotate(
                                        float(neighbor_pad_pos[0]),
                                        float(neighbor_pad_pos[1]),
                                        v.get_orientation())
                                    neighbor_node_pos = v.get_pos()
                                    break

                            p1 = [current_node_pos[0] + rotated_current_pad_pos[0], current_node_pos[1] + rotated_current_pad_pos[1]]
                            p2 = [neighbor_node_pos[0] + rotated_neighbor_pad_pos[0], neighbor_node_pos[1] + rotated_neighbor_pad_pos[1]]

                            lengths.append(np.sqrt(np.square(p1[0]-p2[0])+np.square(p1[1]-p2[1])))

        if len(lengths) != 0: all_lengths.append(np.min(np.array(lengths)))

    return np.sum(all_lengths)

def distance_between_two_points(p1,p2):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        return 0
    else:
        return np.sqrt(np.square(p1[0]-p2[0])+np.square(p1[1]-p2[1]))

# img and los_segment should be layers containing drawings with a single value.
def shortest_distance_to_object_within_segment(img,
                                               los_segment,
                                               centre,
                                               radius,
                                               normalize=True,
                                               padding=(0,0)):
    tmp = los_segment/16 * img/64

    dist = radius
    coords = (-1,-1)

    for i in range(tmp.shape[1]):
        for j in range(tmp.shape[0]):
            if tmp[i][j] == 1:
                d = distance_between_two_points(
                    (centre[0]+padding[0],centre[1]+padding[1]),
                    (j,i))
                if d < dist:
                    dist = d
                    coords = tuple([j,i])

    if normalize:
        dist /= radius

    return dist, coords

def get_coords_from_polar_vector(r, theta, p0, angle_degrees=False):
    if angle_degrees is  True:
        if theta > 180: theta -= 360
        theta = (theta / 180.0) * np.pi

    p = [0,0]

    p[0] = p0[0] + r*np.cos(theta)
    p[1] = p0[1] - r*np.sin(theta)

    return (p[0],p[1]), (p0[0], p0[1])

# orientation invariant
# theta= -theta
def distance_from_rectangle_center_to_edge( size, theta, degrees=True ):
    if degrees is True:
        theta = deg2rad(theta)

    if np.abs(np.tan(theta)) <= (size[1]/size[0]):
        m = 0.5 * (size[0] / np.abs(np.cos(theta)))
    else:
        m = 0.5 * (size[1] / np.abs(np.sin(theta)))

    return m

def deg2rad(theta):
    return (theta / 360) * 2 * np.pi

def rad2deg(theta):
    return (theta * 360) / (2 * np.pi)

def get_los_feature_vector(n, nn, eoi, b, clamp_at_zero=True, padding=None):
    current_node_size = n.get_size()
    current_node_position = n.get_pos()
    current_node_orientation = n.get_orientation()

    los_segments, _ =  draw_los(current_node_position[0],
                                         current_node_position[1],
                                         np.max(current_node_size)*2,
                                         current_node_orientation,
                                         bx=b.get_width(),
                                         by=b.get_height(),
                                         padding=padding)

    grid_comps = draw_comps_from_nodes_and_edges(n,
                                                 nn,
                                                 eoi,
                                                 b,
                                                 padding=padding)

    scaled_current_node_pos = [0,0]
    scaled_current_node_pos[0] = int(current_node_position[0]/pcbDraw_resolution()) + int(3/pcbDraw_resolution())  # centre is offset for padding
    scaled_current_node_pos[1] = int(current_node_position[1]/pcbDraw_resolution()) + int(3/pcbDraw_resolution())  # centre is offset for padding

    los_feature = []
    box_edge_coords = []
    intersection_point_coords = []
    los_radius = int( (np.max(current_node_size)*2) / pcbDraw_resolution())
    for i in range(8):
        # The center point of the current node must factor in the padding of
        # the grid!
        d, c = shortest_distance_to_object_within_segment(
            grid_comps[0],
            los_segments[i],
            tuple([scaled_current_node_pos[0],scaled_current_node_pos[1]]),
            los_radius,
            normalize=False)

        intersection_point_coords.append(c)
        # if distance is less than the radius of the los signal
        if d < los_radius:
            if d == 0:
                angle = current_node_orientation
                d = 1E-3
                m = np.sqrt(np.square(current_node_size[0]/2)+np.square(current_node_size[1]/2)) / pcbDraw_resolution()
                # component center as edge coord
                box_edge_coords.append(
                    (scaled_current_node_pos[0], scaled_current_node_pos[1]))
            else:
                dx = c[0] - scaled_current_node_pos[0]
                dy = c[1] - scaled_current_node_pos[1]
                angle = np.arctan2( (dy+1E-15)  , (dx+1E-15) )

                if angle < 0: angle += 2 * np.pi

                angle = 2 * np.pi - angle

                x = np.int0(
                    np.ceil(current_node_size[0] / pcbDraw_resolution()))
                y = np.int0(
                    np.ceil(current_node_size[1] / pcbDraw_resolution()))
                m = distance_from_rectangle_center_to_edge(
                    (x,y),
                    angle-deg2rad(current_node_orientation),
                    degrees=False )

                box_edge_coords.append(
                    (np.int0(scaled_current_node_pos[0] + m*np.cos(angle)),
                    np.int0(scaled_current_node_pos[1] - m*np.sin(angle)))
                    )

            edge_to_intersection_dist = d - m
        else:
            m=0
            edge_to_intersection_dist = d - 0
            angle = 0
            box_edge_coords.append(
                (scaled_current_node_pos[0], scaled_current_node_pos[1]))

        if clamp_at_zero is True:
            max(edge_to_intersection_dist, 0)

        los_feature.append(edge_to_intersection_dist/los_radius)
    return los_feature, box_edge_coords, intersection_point_coords, {"los_segments":los_segments}

# wraps theta between -pi and pi. Angle always returned in radians
def wrap_angle(theta, degrees=True):
    if degrees is True:
        theta = deg2rad(theta)

    if theta > np.pi:
        return theta - np.pi
    else:
        return theta

def cosine_distance_for_two_terminal_component(resultant, degrees=False):
    """
    vector resultant vector list.
    degrees - When true the angles are interpreted in degrees. By default
    they are interpreted in radians.
    """

    if len(resultant) == 2:
        return np.cos(resultant[0][1][-1] - resultant[1][1][-1])
    else:
        if len(resultant) > 2:
            print("Function 'cosine_distance_for_two_terminal_component' can only work with two resultant vectors. Returning 0.")
        return 0
