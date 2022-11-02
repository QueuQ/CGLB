import torch
from torch.autograd import Variable
import torch.nn.functional as F

def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

def kaiming_normal_init(m):
	if isinstance(m, torch.nn.Conv2d):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, torch.nn.Linear):
		torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class NET(torch.nn.Module):
    """
        A template for implementing new methods for NCGL tasks. The major part for users to care about is the implementation of the function ``observe()``, which is how the implemented NCGL method learns each new task.

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        self.args = args
        self.activation = F.elu

        # setup network
        self.net = model
        self.net.apply(kaiming_normal_init)

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        self.current_task = 0
        self.n_classes = 5
        self.n_known = 0
        self.prev_model = None
 
    def forward(self, g, features):
        
        h = features
        h = self.feature_extractor(g, h)[0]
        if len(h.shape)==3:
            h = h.flatten(1)
        h = self.activation(h)
        h = self.gat(g, h)[0]
        if len(h.shape)==3:
            h = h.mean(1)
        return h
                
    def observe(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
                The method for learning the given tasks. Each time a new task is presented, this function will be called to learn the task. Therefore, how the model adapts to new tasks and prevent forgetting on old tasks are all implemented in this function.
                More detailed comments accompanying the code can be found in the source code of this template in our GitHub repository.

                :param args: Same as the args in __init__().
                :param g: The graph of the current task.
                :param features: Node features of the current task.
                :param labels: Labels of the nodes in the current task.
                :param t: Index of the current task.
                :param prev_model: The model obtained after learning the previous task.
                :param train_ids: The indices of the nodes participating in the training.
                :param ids_per_cls: Indices of the nodes in each class (not in use in the current baseline).
                :param dataset: The entire dataset (not in use in the current baseline).

                """

        # Always set the model in the training mode, since this function will not be called during testing
        self.net.train()
        # if the given task is a new task, set self.current_task to denote the current task index. This is mainly designed for the mini-batch training scenario, in which the data of a task may not come in simultaneously, and each batch of data may either belong to an existing task or a new task..
        if t != self.current_task:            
            self.current_task = t        
            
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t) # Since the output dimensions of a model may correspond to multiple tasks, offset1 and offset2 denote the starting end ending dimension of the output for task t.
        logits = self.net(g, features) # get the model output

        output_labels = labels[train_ids]

        # choose whether to balance the loss with the class sizes
        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))

        # Whether the total number of dimensions are pre-defined or increase with the incoming of new tasks
        if args.classifier_increase:
            loss = self.ce(logits[train_ids, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(logits[train_ids], output_labels, weight=loss_w_)

        if t > 0:
            # This part is for prevent forgetting, the following example is from LwF using knowledge distillation to transfer the previous knowledge to the new model for alleviating forgetting.
            target_ = prev_model.forward(g, features)
            if isinstance(target_, tuple):
                target_ = target_[0]
            target = target_[train_ids]
            o1, o2 = self.task_manager.get_label_offset(t - 1)
            logits_dist = logits[train_ids,o1:o2]
            dist_target = target[:, o1:o2]
            dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, args.lwf_args['T'])
            loss = loss + args.lwf_args['lambda_dist']*dist_loss
        
        # Now the loss contains two parts, one optimizes the model to adapt to the new task, and the other serves to alleviate forgetting.
        loss.backward()
        self.opt.step()
        # no output is required from this function, it only serves to trained the model, and the model is the desired output.
