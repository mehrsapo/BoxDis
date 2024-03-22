from __init__ import * 

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-8)
    normalized_tensor = tensor / norm
    return normalized_tensor



def cal_ct_no_h(forward, adj, device, niter=1000, N = 320):

    eigenimage = normalize(torch.randn(1, 1, N, N)).double().to(device)

    with torch.no_grad():
        for _ in range(niter):
            v = normalize(torch.real(forward(eigenimage)))
            eigenimage = normalize(torch.real(adj(v)))
            v = v.clone()
        

    ct = (torch.sum(eigenimage * torch.real(adj(v)))**2)
    return ct



def cal_ct_with_h(forward, adj, h, device, niter=1000, N=320):

    eigenimage = normalize(torch.randn(1, 1, N, N)).double().to(device)

    with torch.no_grad():
        for _ in range(niter):
            v = normalize(torch.real(forward(eigenimage, h)))
            eigenimage = normalize(torch.real(adj(v, h)))
            v = v.clone()
        

    ct = (torch.sum(eigenimage * torch.real(adj(v, h)))**2)
    return ct
