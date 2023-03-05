import datetime

# parse parameters relevant to the environmnet only.
# do not alter cla in any way. 
class parameters: 
    def __init__(self, pcb_params=None):

        if pcb_params is None:
            print("PCB parameters are needed to initialize the agent. Please provide a dictionary with PCB parameters when creating an agent object.")
            return -1

        self.board_width = pcb_params["board_width"]   
        self.board_height = pcb_params["board_height"]
        self.node = pcb_params["node"]                         # handle to curret node
        self.neighbors = pcb_params["neighbors"]               # list of handles to neighbor nodes
        #self.nets = pcb_params["nets"]                         # list of net ids 
        self.eoi = pcb_params["eoi"]                           # list of edges of interest

        self.sb3_net = pcb_params["net"]                       # path to stable baselines 3 neural network. model extracted trom the file extension.

        self.padding = 4       
    
    def write_to_file(self, fileName, append=True):
        return
    
    def write_to_tensoboard(self, tag):
        return
    
