from __init__ import * 


def create_grid_coords(x_min, x_max, N, device='cpu'): 
    x = torch.linspace(x_min, x_max, N)
    grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')
    g_x = grid_x.flatten()[:, None]
    g_y = grid_y.flatten()[:, None]
    grid_points = torch.cat((g_x, g_y), dim=1).to(device)
    return grid_points


def create_grid_coords_neq(x_min, x_max, N1, N2, device='cpu'): 
    x = torch.linspace(x_min, x_max, N1)
    y = torch.linspace(x_min, x_max, N2)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    g_x = grid_x.flatten()[:, None]
    g_y = grid_y.flatten()[:, None]
    grid_points = torch.cat((g_x, g_y), dim=1).to(device)
    return grid_points