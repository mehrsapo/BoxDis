from __init__ import * 

class prox_tv_Pixel(): 

    def __init__(self, device, box_const):
         
        self.device = device
        self.tv = TV_Pixel(device)
        self.box_const = box_const
    
    def P_c(self, x):
        if self.box_const is None:
            return x
        else:
            return torch.clip(x, self.box_const[0], self.box_const[1])
    
    def eval(self, y, N, niter, lmbda, h, verbose=False, eps=-1, stop=-1):

        v_k = torch.zeros((1, 2, N+2, N+2), requires_grad=False, device=self.device).double()
        u_k = torch.zeros((1, 2, N+2, N+2), requires_grad=False, device=self.device).double()

        t_k = 1 

        alpha = 1 / (8 * (h **2) * lmbda)

        self.loss_list = list()
        loss_old = ((y)**2).sum() / 2
        
        for iters in range(niter):
            LTv = self.tv.Lt(v_k, h)
            x_unproj = y - lmbda * LTv
            x = self.P_c(x_unproj)
            Lpc = self.tv.L(x, h)

            if verbose: 
                loss = ((y - self.P_c(y - lmbda * LTv))**2).sum() / 2 + lmbda * Lpc.abs().sum()
                self.loss_list.append(loss.item())
                
            u_kp1 = torch.clip(v_k + alpha * Lpc, -1, 1)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

            if eps > 0:
                P = 1/2 *((x - y)**2).sum() + lmbda * Lpc.abs().sum()
                Q  = - 1/2 *(x_unproj**2).sum() + 1/2 *((x_unproj - x)**2).sum() + 1/2 *(y**2).sum()
                G_k = P - Q 

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

            if (eps > 0) and (G_k < eps): 
                break
        
        c = y - lmbda * self.tv.Lt(u_k, h)
        c = self.P_c(c)

        return c