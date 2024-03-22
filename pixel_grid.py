import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
import scipy

class PixelGrid(): 

    def __init__(self, N, x_min, x_max, c_init=None, device='cpu'):

        self.device = device
        assert x_min < x_max
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.h = (x_max - x_min) / (N-1) 
        self.a = (N-1) / (x_max - x_min)
        self.b =  - self.a * x_min
        self.lip_H = None
        
        if c_init is not None:
            assert c_init.size(2) == c_init.size(3) == N-1
            self.c_true = c_init
        else:
            self.c_true = torch.zeros((1, 1, N-1, N-1))

        self.cpad = F.pad(self.c_true, (0, 1, 0, 1)).to(device)# c_pad to make the right and down boundaries correct

        return


    def transform_to_grid(self, x):
        return self.a * x + self.b 


    def find_square(self, point): 
        x, y = (point[..., 0:1], point[..., 1:])
        s_x = (x // 1).int()
        s_y = (y // 1).int()

        v1 = torch.cat([s_x, s_y], dim=-1) 
        return v1
    
    def evaluate(self, x): 
        x = self.transform_to_grid(x)
        v1 = self.find_square(x)

        const = self.cpad[0, 0, v1[:, 0].type(torch.LongTensor), v1[:, 1].type(torch.LongTensor)]

        return const
    
    
