from __init__ import * 

class prox_htv_inexact(): 

    def __init__(self, device, box_const=None):
         
        self.device = device
        self.htv = HTV(device)
        self.box_const=box_const


    def P_c(self, x, bc='zero'):
        if bc == 'zero':
            x = F.pad(x[:, :, 1:-1, 1:-1], (1, 1, 1, 1))
        if self.box_const is None:
            return x
        else:
            return torch.clip(x, self.box_const[0], self.box_const[1])
        
    def duality_gap(self, u, y, lmbda):
        x_unproj = y - self.htv.Lt(u)
        x = self.P_c(x_unproj)
        P = 1/2 *((x - y)**2).sum() + lmbda * self.htv.L(x).abs().sum()
        Q  = - 1/2 *(x_unproj**2).sum() + 1/2 *((x_unproj - x)**2).sum() + 1/2 *(y**2).sum()
        G = P - Q 
        return P, Q, G

    def eval(self, y, N, niter, lmbda, eps=1e-8, verbose=False, bc='zero', stop=-1):

        v_k = torch.zeros((1, 3, N+1, N+1), requires_grad=False, device=self.device).double()
        u_k = torch.zeros((1, 3, N+1, N+1), requires_grad=False, device=self.device).double()

        t_k = 1 

        alpha = 1 / 64

        self.loss_list = list()

        loss_old = ((y)**2).sum() / 2
        
        for iters in range(niter):
            LTv = self.htv.Lt(v_k)
            x_unproj = y - LTv
            x = self.P_c(x_unproj, bc)
            Lpc = self.htv.L(x)

            if verbose: 
                loss = ((y - self.P_c(y - LTv, bc))**2).sum() / 2 + lmbda * Lpc.abs().sum()
                self.loss_list.append(loss.item())
                
            u_kp1 = torch.clip(v_k + alpha * Lpc, -lmbda , lmbda)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

            if eps > 0:
                P = 1/2 *((x - y)**2).sum() + lmbda * Lpc.abs().sum()
                Q  = - 1/2 *(x_unproj**2).sum() + 1/2 *((x_unproj - x)**2).sum() + 1/2 *(y**2).sum()
                G_k = P - Q 

            u_k = u_kp1
            v_k = v_kp1
            t_k = t_kp1

            if (eps > 0) and (G_k < eps): 
                break

            if stop > 0 and verbose:
                loss_new = loss
                crit = torch.abs(loss_new - loss_old)
                if (crit < (stop * loss_old)) and (iters > 3):
                    self.niters = iters + 1
                    break
                
                loss_old = loss_new

        c = y - self.htv.Lt(u_k)
        c = self.P_c(c, bc)

        return c
