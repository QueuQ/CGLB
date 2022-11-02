import torch.nn as nn

class PairNorm(nn.Module):
    def __init__(self, mode='PN-SCS', scale=1.0):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
        
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class DynamicPairNorm(nn.Module):
    """
    generate the scale and mean for pair norm operation
    the scale and mean is different for every node, like the layer norm
    Args:
        graph contains both the edges and feature of node
    Output:
        the scale and mean for each node
    """
    def __init__(self):
        super(DynamicPairNorm, self).__init__()
    
    def __TransFeauture(self):
        """
        translate the feature of node to pair distance, so that network can be used for any
        dimmensions.
        TODO:
            what if the input graph already hold edge feature?
        Args:
            graph with any features
        Output:
            graph without node's features but with edge features
        """

