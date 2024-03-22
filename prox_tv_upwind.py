from __init__ import * 

class prox_tv_upwind(): 

    def __init__(self, device, box_const):
         
        self.device = device
        self.tv = TV_upwind(device)
        self.box_const = box_const
    
    def P_c(self, x):
        if self.box_const is None:
            return x
        else:
            return torch.clip(x, self.box_const[0], self.box_const[1])

    
    def eval(self, y, N, niter, lmbda, h, verbose=False, stop=-1):

        v_k = torch.zeros((1, 4, N+2, N+2), requires_grad=False, device=self.device).double()
        u_k = torch.zeros((1, 4, N+2, N+2), requires_grad=False, device=self.device).double()

        t_k = 1 

        alpha = 1 / (16 * (h **2) * lmbda)

        self.loss_list = list()

        loss_old = ((y)**2).sum() / 2
        
        for iters in range(niter):
            LTv = self.tv.Lt(v_k, h)
            Lpc = self.tv.L(self.P_c(y - lmbda * LTv), h)

            if verbose: 
                loss = ((y - self.P_c(y - lmbda * LTv))**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=2).sum()
                self.loss_list.append(loss.item())
                
            u_kp1 = F.normalize(F.relu(v_k + alpha * Lpc), eps=1, dim=1, p=2)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

            u_k = u_kp1
            v_k = v_kp1
            t_k = t_kp1

            if stop > 0 and verbose:
                loss_new = loss
                crit = torch.abs(loss_new - loss_old)
                if (crit < (stop * loss_old)) and (iters > 3):
                    self.niters = iters + 1
                    break
                
                loss_old = loss_new

        c = y - lmbda * self.tv.Lt(u_k, h)
        c = self.P_c(c)

        return c
