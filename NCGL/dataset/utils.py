import collections

class semi_task_manager:
    def __init__(self):
        self.task_info = collections.OrderedDict()
        self.task_info[-1] = 0
        self.g = None
        self.newg = []
        self.degree = None

    def add_task(self, task_i, masks):
        self.task_info[task_i] = masks

    def retrieve_task(self, task_i):
        """Retrieve information for task_i
        """
        return self.task_info[task_i]

    def old_tasks(self):
        return self.task_info.keys()

    def get_label_offset(self, task_i, original=False):
        if original:
            if task_i>0:
                return self.task_info[task_i-1], self.task_info[task_i]
            else:
                return 0, self.task_info[task_i]
        else:
            return 0, self.task_info[task_i]

    def add_g(self, g):
        self.g  = g
    
    def add_newg(self, newg):
        self.newg.append(newg)
        
    def add_degree(self, degree):
        self.degree = degree

class task_manager:
    def __init__(self):
        pass