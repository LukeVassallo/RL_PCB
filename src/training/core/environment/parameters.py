# parse parameters relevant to the environmnet only.
# do not alter cla in any way.
class parameters:
    def __init__(self, params=None):
        if params is None:
            print("params are needed to initialize the environment.\
                   Please provide a dictionary with PCB params when\
                   creating an agent object.")
            return -1
        self.training_pcb = params["training_pcb"]
        self.evaluation_pcb = params["evaluation_pcb"]
        self.pcb_file = params["pcb_file"]
        self.net = params["net"]
        self.use_dataAugmenter = params["use_dataAugmenter"]
        self.augment_position = params["augment_position"]
        self.augment_orientation = params["augment_orientation"]
        self.agent_max_action = params["agent_max_action"]
        self.agent_expl_noise = params["agent_expl_noise"]
        self.debug = params["debug"]
        self.max_steps = params["max_steps"]
        # weight for euclidean distance based wirelength
        self.n = params["w"]
        # weight for overlap
        self.m = params["o"]
        # weight for hpwl
        self.p = params["hpwl"]
        self.seed = params["seed"]
        self.ignore_power = params["ignore_power"]
        self.log_dir = params["log_dir"]
        # TODO: Add error checking
        self.idx = params["idx"]
        self.shuffle_idxs = params["shuffle_idxs"]
    def write_to_file(self, fileName, append=True):
        return

    def write_to_tensoboard(self, tag):
        return

    def to_string(self):
        params = vars(self)
        s = ""
        s += "<strong>====== Environment parameters ======</strong><br>"
        for key,value in params.items():
            s += f"{key} -> {value}<br>"
        s += "<br>"

        return s

    def to_text_string(self, prefix = ""):
        params = vars(self)
        s = ""
        for key,value in params.items():
            s += f"{prefix}{key} -> {value}\r\n"

        return s
