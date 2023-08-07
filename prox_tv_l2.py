from __init__ import * 

class prox_tv_l2(): 

    def __init__(self, device):
         
        self.device = device
        self.tv = TV_l2(device)

    
    def eval(self, y, N, niter, lmbda, h, verbose=False, toi=1e-4):

        v_k = torch.zeros((2, 2, N+2, N+2), requires_grad=False, device=self.device).double()
        u_k = torch.zeros((2, 2, N+2, N+2), requires_grad=False, device=self.device).double()

        t_k = 1 

        alpha = 1 / (4 * lmbda)

        self.loss_list = list()

        for _ in range(niter):
            LTv = self.tv.Lt(v_k, h)
            Lpc = self.tv.L(torch.clip(y - lmbda * LTv, 0, 1), h)

            if verbose: 
                loss = ((y - torch.clip(y - lmbda * LTv, 0, 1))**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=2).sum()
                self.loss_list.append(loss.item())
                
            u_kp1 = F.normalize(v_k + alpha * Lpc, eps=1, dim=1, p=2)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

            err_rel = torch.norm(u_k - u_kp1, p='fro') / max((torch.norm(u_k, p='fro')).item(), 1e-10)

            u_k = u_kp1
            v_k = v_kp1
            t_k = t_kp1

            if err_rel < toi: 
                break

        c = y - lmbda * self.tv.Lt(u_k, h)
        c = torch.clip(c, 0, 1)

        return c
