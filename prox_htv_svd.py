from __init__ import * 

class prox_htv_svd(): 

    def __init__(self, device, box_const=None):
         
        self.device = device
        self.htv = HTV_svd_closed(device)
        self.box_const=box_const


    def schatten_norm_2d_closed(self, hessian, im_size_x, im_size_y, device):

        ii, ij, ji, jj = (0, 2, 2, 1)

        tr = hessian[0, ii, ...] + hessian[0, jj, ...]
        dt = (hessian[0, ii, ...] - hessian[0, jj, ...])**2 + 4*hessian[0, ij, ...]*hessian[0, ji, ...]

        singvals = torch.zeros([2, 1, im_size_x, im_size_y]).to(device).double()
        singvals[0, ...] = 0.5*(tr+torch.sqrt(dt))
        singvals[1, ...] = 0.5*(tr-torch.sqrt(dt))

        norm = torch.norm(singvals, p=1).sum()

        return norm

    def schatten_proj_2d_closed(self, hessian, im_size_x, im_size_y, device):

        ii, ij, ji, jj = (0, 2, 2, 1)

        tr = hessian[0, ii, ...] + hessian[0, jj, ...]
        dt = (hessian[0, ii, ...] - hessian[0, jj, ...])**2 + 4*hessian[0, ij, ...]*hessian[0, ji, ...]

        singvals = torch.zeros([2, 1, im_size_x, im_size_y]).to(device).double()
        singvals[0, ...] = 0.5*(tr+torch.sqrt(dt))
        singvals[1, ...] = 0.5*(tr-torch.sqrt(dt))

        singvec = torch.zeros([2, 1, im_size_x, im_size_y]).to(device).double()
        eps = 2.*torch.finfo(float).eps
        zero_ji = (torch.abs(hessian[0, ji, ...])<eps) * 1
        singvec[0, ...] = (1-zero_ji)*(singvals[0, ...] - hessian[0, jj, ...])
        singvec[1, ...] = (1-zero_ji)*hessian[0, ji, ...] 

        norm = torch.sqrt((singvals[0, ...] - hessian[0, jj, ...])**2 + hessian[0, ji, ...]**2) + zero_ji
        singvec[0, ...] = singvec[0, ...]/norm + zero_ji
        singvec[1, ...] /= norm

        # project 
        singvals_prj = torch.clip(singvals, -1, 1)

        # rebuild svd
        s = singvals_prj
        V = singvec
        hessian_rebuild = torch.zeros([1, 3, im_size_x, im_size_y]).to(device).double()
        hessian_rebuild[0, 0, ...] = s[0]*V[0]**2 + s[1]*V[1]**2
        hessian_rebuild[0, 2, ...] = (s[0]-s[1])*V[0]*V[1]
        hessian_rebuild[0, 1, ...] = s[0]*V[1]**2 + s[1]*V[0]**2
        
        return hessian_rebuild
    
    def eval(self, y, N, niter, lmbda, eps=-1, verbose=False, bc='zero', stop=-1):


        self.loss_list = list()

        omg_k = torch.zeros((1, 3,N, N), requires_grad=False, device=self.device).double()
        psi_k = torch.zeros((1, 3,N, N), requires_grad=False, device=self.device).double()
        t_k = 1 

        alpha = 1/(64*lmbda)
        loss_old = ((y)**2).sum() / 2

        for iters in range(niter):
            pc = torch.clip(y - lmbda * self.htv.hessian_adj_2d_closed(psi_k), 0, 1)
            Lpc = self.htv.hessian_2d_closed(pc)
            omg_kp1 = self.schatten_proj_2d_closed(psi_k + alpha * Lpc, N, N, self.device)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            psi_kp1 = omg_k + (t_k - 1) / t_kp1 * (omg_kp1 - omg_k)                                                                                                                                                                                                                                                                                                                                         

            if verbose: 
                loss = ((y - pc)**2).sum() / 2 + lmbda * self.schatten_norm_2d_closed(Lpc, N, N, self.device)
                self.loss_list.append(loss.item())

            omg_k = omg_kp1
            psi_k = psi_kp1
            t_k = t_kp1

            if stop > 0 and verbose:
                loss_new = loss
                crit = torch.abs(loss_new - loss_old)
                if (crit < (stop * loss_old)) and (iters > 3):
                    self.niters = iters + 1
                    break
            
                loss_old = loss_new

    

        c = torch.clip(y - lmbda * self.htv.hessian_adj_2d_closed(omg_k), 0, 1)

        return c
