from __init__ import * 

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-8)
    normalized_tensor = tensor / norm
    return normalized_tensor


device  = 'cuda:0'
prox_tv_l2_ = prox_tv_l2(device)
eigenimage = normalize(torch.randn(1, 1, 320, 320)).double().to(device)

with torch.no_grad():
    for i in range(10000):
        v = normalize(torch.real(prox_tv_l2_.tv.L(eigenimage, 1)))
        eigenimage = normalize(torch.real(prox_tv_l2_.tv.Lt(v, 1)))
        v = v.clone()
        

ct = (torch.sum(eigenimage * torch.real(prox_tv_l2_.tv.Lt(v, 1)))**2)
print('ct tv_l2: ', ct)

prox_tv_l1_ = prox_tv_l1(device)
eigenimage = normalize(torch.randn(1, 1, 320, 320)).double().to(device)

with torch.no_grad():
    for i in range(10000):
        v = normalize(torch.real(prox_tv_l1_.tv.L(eigenimage, 1)))
        eigenimage = normalize(torch.real(prox_tv_l1_.tv.Lt(v, 1)))
        v = v.clone()
        

ct = (torch.sum(eigenimage * torch.real(prox_tv_l1_.tv.Lt(v, 1)))**2)
print('ct tv_l1: ', ct)

prox_htv_ = prox_htv(device)
eigenimage = normalize(torch.randn(1, 1, 320, 320)).double().to(device)

with torch.no_grad():
    for i in range(10000):
        v = normalize(torch.real(prox_htv_.htv.L(eigenimage)))
        eigenimage = normalize(torch.real(prox_htv_.htv.Lt(v)))
        v = v.clone()
        

ct = (torch.sum(eigenimage * torch.real(prox_htv_.htv.Lt(v)))**2)
print('ct htv: ', ct)