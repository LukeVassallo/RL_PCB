# parse parameters relevant to the environmnet only.
# do not alter cla in any way.
class parameters:
    def __init__(self, pcb_params=None):

        if pcb_params is None:
            print("PCB parameters are needed to initialize the agent.\
                   Please provide a dictionary with PCB parameters when\
                   creating an agent object.")
            return -1

        self.board = pcb_params["board"]
        self.graph = pcb_params["graph"]
        self.board_width = pcb_params["board_width"]
        self.board_height = pcb_params["board_height"]
        # handle to curret node
        self.node = pcb_params["node"]
        # list of handles to neighbor nodes
        self.neighbors = pcb_params["neighbors"]
        # list of edges of interest
        self.eoi = pcb_params["eoi"]

        # path to stable baselines 3 neural network.
        # model extracted trom the file extension.
        self.net = pcb_params["net"]
        self.padding = 4
        self.step_size = pcb_params["step_size"]
        self.max_steps = pcb_params["max_steps"]
        self.ignore_power_nets = True
        self.opt_euclidean_distance = pcb_params["opt_euclidean_distance"]
        self.opt_hpwl = pcb_params["opt_hpwl"]
        self.seed = int(pcb_params["seed"])
        self.nets = pcb_params["nets"]
        self.graph = pcb_params["graph"]
        self.max_action = pcb_params["max_action"]
        self.expl_noise = pcb_params["expl_noise"]
        self.n = pcb_params["n"]
        self.m = pcb_params["m"]
        self.p = pcb_params["p"]
        self.ignore_power = pcb_params["ignore_power"]
        self.log_file = pcb_params["log_file"]

    def write_to_file(self, fileName, append=True):
        return

    def write_to_tensoboard(self, tag):
        return

    def to_string(self):
        params = vars(self)
        s = ""
        s += f"<strong>====== Agent {self.node.get_name()} ({self.node.get_id()}) parameters ======</strong><br>"
        for key,value in params.items():
            if key in ("board", "graph", "node", "neighbors", "eoi", "edge"):
                continue
            s += f"{key} -> {value}<br>"
        s += "<br>"

        return s
