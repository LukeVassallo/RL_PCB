"""Module for parsing command line arguments"""
import argparse
import cv2
import numpy as np
from pcb import pcb
import sys
sys.path.append('../training/')
from pcbDraw import draw_board_from_board_and_graph_with_debug, draw_ratsnest_with_board

def command_line_args():
    """
    Parses command-line arguments for a Python script that generates a PNG image
    from a .pcb file.

    Returns:
    args: argparse.Namespace object containing parsed command-line arguments.
    settings: dict object containing settings for the script. Currently, it
    contains a single key-value pair with the .pcb file path.

    Usage:
    Call this function in your script to parse command-line arguments. The
    function requires that the -p/--pcb option is passed, which should be
    followed by the path to the .pcb file to be converted to a PNG image.
    Example:

    args, settings = command_line_args()
    pcb_file = settings["pcb"]
    """

    parser = argparse.ArgumentParser(description='Python script for generating\
             a .png image from a .pcb file.', usage='python pcb2png.py -p \
            <pcb_file>')

    parser.add_argument('-p', '--pcb', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()
    settings = {}

    settings['pcb'] = args.pcb
    settings['output'] = args.output

    return args, settings

def main():
    _, settings = command_line_args()
    pv = pcb.vptr_pcbs()
    pcb.read_pcb_file(settings['pcb'], pv)      # Read pcb file
    p = pv[0]

    g = p.get_graph()
    g.reset()
    b = p.get_board()
    g.set_component_origin_to_zero(b)

    comp_grids = draw_board_from_board_and_graph_with_debug(b, g, padding=0.5)

    ratsnest = None
    nn = g.get_nodes()
    for i in range(len(nn)):
        node_id = nn[i].get_id()
        nets = []

        neighbor_ids = g.get_neighbor_node_ids(node_id)
        neighbors = []
        for n_id in neighbor_ids:
            neighbors.append(g.get_node_by_id(n_id))

        ee = g.get_edges()
        eoi = []
        for e in ee:
            if e.get_instance_id(0) == node_id or e.get_instance_id(1) == node_id:
                eoi.append(e)
                nets.append(e.get_net_id)

        if i == 0:
            ratsnest = draw_ratsnest_with_board(nn[i], neighbors, eoi, b,
                                          line_thickness=1, padding=0.5,
                                          ignore_power=True)
        else:
            ratsnest = np.maximum(ratsnest,
                            draw_ratsnest_with_board(nn[i], neighbors, eoi, b,
                                                     line_thickness=1,
                                                     padding=0.5,
                                                     ignore_power=True)
                            )

    img = comp_grids[0] + 2*comp_grids[1]
    img = np.maximum(img, ratsnest)
    cv2.imwrite(settings['output'], img)

if __name__ == '__main__':
    main()
