# parse parameters relevant to the environmnet only.
# do not alter cla in any way. 
class parameters: 
    def __init__(self, parameters=None):
        if parameters is None:
            print("Parameters are needed to initialize the environment. Please provide a dictionary with PCB parameters when creating an agent object.")
            return -1
        self.training_pcb = parameters["training_pcb"]
        self.evaluation_pcb = parameters["evaluation_pcb"]
        self.pcb_file = parameters["pcb_file"]        
        self.net = parameters["net"]
        self.use_dataAugmenter = parameters["use_dataAugmenter"]
        self.augment_position = parameters["augment_position"]
        self.augment_orientation = parameters["augment_orientation"]
        self.agent_max_action = parameters["agent_max_action"]
        self.agent_expl_noise = parameters["agent_expl_noise"]
        self.debug = parameters["debug"]
        self.max_steps = parameters["max_steps"]
        self.n = parameters["w"]                   # weight for euclidean distance based wirelength
        self.m = parameters["o"]                    # weight for overlap
        self.p = parameters["hpwl"]                 # weight for hpwl
        self.seed = parameters["seed"]
        self.ignore_power = parameters["ignore_power"]
        self.log_dir = parameters["log_dir"]
        self.idx = parameters["idx"]                # TODO: Add error checking
        self.shuffle_idxs = parameters["shuffle_idxs"]
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
    
    def to_text_string(self, prefix = ''):
        params = vars(self)
        s = ""
        for key,value in params.items():
            s += f"{prefix}{key} -> {value}\r\n"
                    
        return s
    
