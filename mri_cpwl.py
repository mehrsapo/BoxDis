from __init__ import * 

class MRICPWL(): 

    def __init__(self, mask, device) -> None:
    
        self.mask = mask
        self.N = self.mask.size(2)
        self.device = device
        self.sam = SamplerCPWL(device)
        self.h = 1

        return

    def set_h(self, h): 
        self.h = h
        self.cal_box_corr()
        return

    def cal_box_corr(self): 
        N = self.N
        h = self.h 

        if h >= 1: 
            self.w_1 = torch.fft.fftfreq(N).repeat(N, 1) # - 
            self.w_2 = torch.fft.fftfreq(N)[:, None].repeat(1, N)
            w_sum = -self.w_1 + self.w_2 # remove - 
            self.box_corr = (torch.sinc(self.w_1) * torch.sinc(self.w_2) * torch.sinc(w_sum))[None, None, :, :].double().to(self.device) 

        elif h < 1: 
            self.w_22 = torch.fft.fftfreq(int(N/h)).repeat(int(N/h), 1) # -1 
            self.w_12 = torch.fft.fftfreq(int(N/h))[:, None].repeat(1, int(N/h))
            w_sum2 = -self.w_22 + self.w_12 # remove -1
            self.box_corr  = (torch.sinc(self.w_12) * torch.sinc(self.w_22) * torch.sinc(w_sum2))[None, None, :, :].double().to(self.device)

        return

    def H(self, x): 
    
        h = self.h 
        N = self.N
        s = int(np.log2(h))
        if h >= 1: 
            x_up = x
            for _ in range(s): 
                x_up = self.sam.upsample(F.pad(x_up, (0, 1, 0, 1)))[:, :, :-1, :-1]
            Hx = torch.fft.fft2(x_up, norm='ortho') * self.box_corr 

        if h < 1: 
            N_t = N*int(1/h/2)
            N_h = int(N//2)
            c_hat = torch.fft.fft2(x, norm='ortho')
            Hx = torch.fft.ifftshift((torch.fft.fftshift(c_hat * self.box_corr))[:, :, N_t-N_h:N_t+N_h, N_t-N_h:N_t+N_h] * h)

        return Hx * self.mask


    def Ht(self, y): 

        h = self.h 
        N = self.N 
        s = int(np.log2(h))

        y = y * self.mask
        
        if h >= 1: 
            y_down = torch.real(torch.fft.ifft2(y * self.box_corr, norm='ortho'))
            for _ in range(s): 
                y_down = self.sam.downsample(F.pad(y_down, (0, 1, 0, 1)))[:, :, :-1, :-1]

            Hty = y_down
            

        if h < 1: 
            N_t = N*int(1/h/2)
            N_h = int(N//2)

            Hty = torch.zeros((1, 1, N*int(1/h), N*int(1/h))).to(self.device) + 1j * torch.zeros((1, 1, N*int(1/h), N*int(1/h))).to(self.device)
            Hty[:, :, N_t-N_h:N_t+N_h, N_t-N_h:N_t+N_h] = torch.fft.fftshift(y) * h
            Hty = torch.fft.ifftshift(Hty) * self.box_corr
            Hty = torch.fft.ifft2(Hty, norm='ortho') 
        
        return torch.real(Hty)
    


    def prox_G_cg(self, u, y, tau, max_iter=2000, rel_tol=1e-20):
        # with conjugate-gradient inversion
        A = lambda xx: (tau * self.Ht(self.H(xx)) + xx)
        b = (u + tau * self.Ht(y))
        return cg(A, b, u, max_iter, rel_tol)
    

    def prox_G_cf(self, u, y, tau):
        # with closed-form inversion
        b = (u + tau * self.Ht(y))

        h = self.h 
        N = self.N 
        
        N_t = N*int(1/h/2)
        N_h = int(N//2)
        
        if self.h == 1:
            F = torch.fft.fft2(b, norm='ortho') 
            F = F / (1 + tau * (self.box_corr**2) * (self.mask**2))
            out = torch.real(torch.fft.ifft2(F, norm='ortho'))
        elif self.h < 1:
            F = torch.fft.fft2(b, norm='ortho') 
            mask_big = torch.zeros((1, 1, N*int(1/h), N*int(1/h))).to(self.device) 
            mask_big[:, :, N_t-N_h:N_t+N_h, N_t-N_h:N_t+N_h] = torch.fft.fftshift(self.mask)
            mask_big = torch.fft.ifftshift(mask_big)
            F = F / (1 + tau * (self.box_corr**2) * (mask_big**2) * (h**2))
            out = torch.real(torch.fft.ifft2(F, norm='ortho'))
        else:
            out = self.prox_G_cg(u, y, tau)

        return out