from __init__ import * 

class MultiResSolver():

    # only fourier measurements

    def __init__(self, forward_model='fm', reg='htv', lmbda=0., mask=None, h_init=4, range_r=512, N_scales=6,
                  n_iters_in=None, n_iters_out=None, verbose=True, device='cuda:3', toi=1e-4, box_const=None) -> None:
        
        self.forward_model = forward_model
        self.reg = reg
        self.h_init = h_init
        self.N_scales = N_scales
        self.verbose = verbose
        self.lmbda = lmbda
        self.device = device
        self.toi = toi
        self.box_const = box_const

        if  self.forward_model == 'fm':
            if mask is None:
                assert False 
            
            self.mri = MRICPWL(mask, self.device)
            self.N_y = mask.size(2)

            self.h_coeff = range_r / (self.N_y)

            self.N_init = int(range_r / self.h_init) + 1
            self.h_scales = [self.h_init / (2**i) for i in range(self.N_scales)]
            self.size_scales = [self.N_init * 2**(i) - (2**i-1) for i in range(N_scales)]
            self.alphas = [(self.h_coeff**2)/ ((self.h_scales[i])**2) for i in range(self.N_scales)]

        else:
            assert False

        if n_iters_in is None:
            self.n_iters_in = [2000]* self.N_scales
        else: 
            self.n_iters_in = n_iters_in
        
        if n_iters_out is None:
            self.n_iters_out = [2000]* self.N_scales
        else:
            self.n_iters_out = n_iters_out

        self.sols = list()
        self.costs = list()
        self.iters = list()
        self.sam = SamplerCPWL(self.device) 

        if self.reg == 'htv':
            self.prox_htv = prox_htv_inexact(self.device, box_const)

        if self.reg == 'tv_l2':
            self.prox_tv_l2 = prox_tv_l2_inexact(device, box_const)
        
        if self.reg == 'tv_l1':
            self.prox_tv_l1 = prox_tv_l1_inexact(device, box_const)

    


        self.mses = list()
        self.reg_values = list()
        self.losses = list()
        self.n_iters_in_list = list()
        self.n_iters_out_list = list()
        self.times = list()
        
        return
    
    def P_c(self, x):
        x = F.pad(x[:, :, 1:-1, 1:-1], (1, 1, 1, 1))
        if self.box_const is None:
            return x
        else:
            return torch.clip(x, self.box_const[0], self.box_const[1])
        
    def bound_set_fm(self, x): 
        return x[:, :, :-1, :-1]

    def bound_adj_set_fm(self, y):
        return F.pad(y, (0, 1, 0, 1))
        

    def cal_cost_fm(self, y, c_k, n_scale): 
        mse = (torch.abs(y - (self.mri.H(self.bound_set_fm(c_k))))**2).sum() / 2
        if self.reg == 'htv':
            reg_val = self.prox_htv.htv.L(c_k).abs().sum()
        if self.reg == 'tv_l2':
            reg_val = torch.norm(self.prox_tv_l2.tv.L(c_k, self.h_scales[n_scale]), dim=1, p=2).sum()
        if self.reg == 'tv_l1':
            reg_val = torch.norm(self.prox_tv_l1.tv.L(c_k, self.h_scales[n_scale]), dim=1, p=1).sum()

        loss = mse + self.lmbda * reg_val

        return mse, reg_val, loss

    def solve_sacle_fm(self, n_scale, c_k, d_k, y):
        
        n_iter_out = self.n_iters_out[n_scale]
        alpha = self.alphas[n_scale]
        h = self.h_scales[n_scale]

        if h >=1 : 
            h = int(h)

        self.mri.set_h(h /  self.h_coeff)

        t_k = 1
        verbose = self.verbose

        loss_scale = list()
        reg_scale = list()
        mse_scale = list()            

        for i in range(n_iter_out): 
            
            if verbose and (i%1==0 or i==n_iter_out-1) : 
                mse, reg_val, loss = self.cal_cost_fm(y, c_k, n_scale)   

                mse_scale.append(mse.item())
                reg_scale.append(reg_val.item())
                loss_scale.append(loss.item())

                print('iter ' + str(i) + ', loss:' + str(np.round(loss.item(), 8)))

            w_k = d_k + alpha * torch.real(self.bound_adj_set_fm(self.mri.Ht(y - self.mri.H(self.bound_set_fm(d_k)))))
            
            eps  = np.minimum(1e-3, 1 / (i+1)**3)

            if self.reg == 'htv':                
                c_kp1 = self.prox_htv.eval(w_k, self.size_scales[n_scale], self.n_iters_in[n_scale], self.lmbda*alpha, eps=eps)

            if self.reg == 'tv_l2':
                if self.lmbda > 0:
                    c_kp1 = self.prox_tv_l2.eval(w_k, self.size_scales[n_scale], self.n_iters_in[n_scale],
                                              self.lmbda*alpha, self.h_scales[n_scale], eps=eps)
                else:
                    c_kp1 = self.P_c(w_k)

            if self.reg == 'tv_l1':
                if self.lmbda > 0:
                    c_kp1 = self.prox_tv_l1.eval(w_k, self.size_scales[n_scale], self.n_iters_in[n_scale],
                                              self.lmbda*alpha, self.h_scales[n_scale], eps=eps)
                else:
                    c_kp1 = self.P_c(w_k)
        
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            d_kp1 = c_kp1  + (t_k - 1) / t_kp1 * (c_kp1 - c_k)                                                                                                                                                                                                                                                                                                                                                         
            
            if self.toi > 0:
                _, _, l_c_k = self.cal_cost_fm(y, c_k, n_scale)   
                _, _, l_c_kp1 = self.cal_cost_fm(y, c_kp1, n_scale)   
                e_k = torch.abs(l_c_k - l_c_kp1)

            c_k = c_kp1
            d_k = d_kp1
            t_k = t_kp1

            if (i > 25) and (self.toi > 0) and (e_k <= (self.toi * l_c_k)): 
                break

        c_hat = self.P_c(c_k)

        mse, reg_val, cost = self.cal_cost_fm(y, c_hat, n_scale)

        if verbose:
            print('final loss : ' + str(np.round(cost.item(), 8)))
            self.mses.append(mse_scale)
            self.reg_values.append(reg_scale)
            self.losses.append(loss_scale)

        
        return i+1, c_hat, cost
    
    
    def solve_fm(self, y):
        c_k = torch.zeros((1, 1, self.size_scales[0], self.size_scales[0])).double().to(self.device)
        d_k = torch.zeros((1, 1, self.size_scales[0], self.size_scales[0])).double().to(self.device)

        self.iterand = 0
        for n_scale in range(self.N_scales):
            if self.verbose:
                print('----------- h = ' + str(self.h_scales[n_scale]) + ' -----------' )
            
            t0 = time.time()
            iters, c_hat, cost = self.solve_sacle_fm(n_scale, c_k, d_k, y)
            t1 = time.time()

            self.iters.append(iters)
            self.sols.append(c_hat)
            self.costs.append(cost.cpu().item())
            self.times.append(t1-t0)
            
            c_k = self.sam.upsample(c_hat)
            d_k = self.sam.upsample(c_hat)


    
if __name__ == '__main__':

    pass
    

