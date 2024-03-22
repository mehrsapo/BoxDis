from __init__ import  * 



class BoxGrid(): 

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
            assert c_init.size(2) == c_init.size(3) == N
            self.c_true = c_init
        else:
            self.c_true = torch.zeros((1, 1, N, N))

        self.cpad = F.pad(self.c_true, (0, 1, 0, 1)).to(device)# c_pad to make the right and down boundaries correct

        return


    def transform_to_grid(self, x):
        return self.a * x + self.b 


    def find_simplex(self, point): 
        x, y = (point[..., 0:1], point[..., 1:])
        s_x = (x // 1).int()
        s_y = (y // 1).int()
        centerd_point_x, centerd_point_y = (x - s_x, y - s_y)

        v1, v2 = (torch.cat([s_x + 1, s_y], dim=-1) , torch.cat([s_x, s_y+1], dim=-1))
        v3 = torch.where(centerd_point_x >=  1 - centerd_point_y, torch.cat([s_x + 1, s_y + 1], dim=-1), torch.cat([s_x, s_y], dim=-1))
    

        return v1, v2, v3

    def basis_eval(self, x): 
        x[:, 0] = x[:, 0] * -1   # to correct for the geometric coordinate and the image coordinate
        x = F.pad(x, (0, 1))

        min_ = torch.min(x, dim=1).values
        max_ = torch.max(x, dim=1).values

        phi = F.relu(1 + min_ - max_)
        return phi
    
    def evaluate(self, x): 
        x = self.transform_to_grid(x)
        v1, v2, v3 = self.find_simplex(x)

        phi_1 = self.basis_eval(x - v1)
        phi_2 = self.basis_eval(x - v2)
        phi_3 = self.basis_eval(x - v3)

        c_1 = self.cpad[0, 0, v1[:, 0].type(torch.LongTensor), v1[:, 1].type(torch.LongTensor)]
        c_2 = self.cpad[0, 0, v2[:, 0].type(torch.LongTensor), v2[:, 1].type(torch.LongTensor)]
        c_3 = self.cpad[0, 0, v3[:, 0].type(torch.LongTensor), v3[:, 1].type(torch.LongTensor)]

        output = c_1 * phi_1 + c_2 * phi_2 + c_3 * phi_3

        return output
    
    def construct_mat(self, x):

        x = self.transform_to_grid(x)

        v1, v2, v3 = self.find_simplex(x)

        phi_1 = self.basis_eval(x - v1)
        phi_2 = self.basis_eval(x - v2)
        phi_3 = self.basis_eval(x - v3)

        idx1 = v1[:, 1:].type(torch.LongTensor) + (self.N+1) * v1[:, :1].type(torch.LongTensor)
        idx2 = v2[:, 1:].type(torch.LongTensor) + (self.N+1) * v2[:, :1].type(torch.LongTensor)
        idx3 = v3[:, 1:].type(torch.LongTensor) + (self.N+1) * v3[:, :1].type(torch.LongTensor)

        grid_points_of_simplices_of_points = np.concatenate((idx1, idx2, idx3), axis=1)

        basis_values = torch.cat((phi_1[:, None], phi_2[:, None], phi_3[:, None]), axis=1)

        rows = np.repeat(np.arange(x.size(0)), 3)

        cols = grid_points_of_simplices_of_points.flatten()
        data = basis_values.flatten()

        H = csr_matrix((data, (rows, cols)), shape=(x.size(0), (self.N+1)**2), dtype='float64')
        H.eliminate_zeros()
        H.check_format()

        _, s, _ = scipy.sparse.linalg.svds(H, k=1)
        self.lip_H = s[0] ** 2 

        HT = H.transpose().tocsr()

        self.H_mat = torch.sparse_csr_tensor(H.indptr.tolist(), H.indices.tolist(), H.data.tolist(), dtype=torch.double, size = H.shape, device=self.device)
        self.HT_mat  = torch.sparse_csr_tensor(HT.indptr.tolist(), HT.indices.tolist(), HT.data.tolist(), dtype=torch.double, device=self.device)

        
        return
    


    def H(self, c):
        return self.H_mat @ F.pad(c, (0, 1, 0, 1)).view((self.N+1)**2, 1)
    

    def Ht(self, y):
        return (self.HT_mat @ y).view(1, 1, self.N+1, self.N+1)[:, :, :-1, :-1]
