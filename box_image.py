from __init__ import *

def create_grid_coords(N, n, device): 
    x = torch.linspace(0, N-1, n)
    grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')
    g_x = grid_x.flatten()[:, None]
    g_y = grid_y.flatten()[:, None]
    grid_points = torch.cat((g_x, g_y), dim=1).to(device)
    return grid_points
    
class BoxImage(): 

    def __init__(self, c):

        self.N = c.size(3)
        self.c = F.pad(c, (0, 1, 0, 1))

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

        v1, v2, v3 = self.find_simplex(x)

        phi_1 = self.basis_eval(x - v1)
        phi_2 = self.basis_eval(x - v2)
        phi_3 = self.basis_eval(x - v3)

        c_1 = self.c[0, 0, v1[:, 0].type(torch.LongTensor), v1[:, 1].type(torch.LongTensor)]
        c_2 = self.c[0, 0, v2[:, 0].type(torch.LongTensor), v2[:, 1].type(torch.LongTensor)]
        c_3 = self.c[0, 0, v3[:, 0].type(torch.LongTensor), v3[:, 1].type(torch.LongTensor)]

        output = c_1 * phi_1 + c_2 * phi_2 + c_3 * phi_3

        return output
        

if __name__ == "__main__": 

    device = 'cpu'
    c1 = torch.Tensor([[[[4, 7, 12], [1, 3, 2], [6, 4, 3]]]]).to(device).double() 

    cont_image = BoxImage(c1, 1)

    new_point = torch.Tensor([[0.5, 0.25], [0.25, 0.5]])

    values = cont_image.evaluate(new_point)

    print(values)