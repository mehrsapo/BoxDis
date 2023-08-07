from __init__ import *

torch.pi = torch.acos(torch.zeros(1)).item() * 2 

# --------------------------------------------- 2d prox htv svd  -----------------------------------------------

def schatten_norm_2d(hessian, im_size_x, im_size_y, device):

    a, b, c = hessian[: ,0], hessian[: ,2], hessian[:,1]
    mat = torch.cat((a, b, b, c), dim=0).view(2, 2, -1).permute(2, 0, 1)
    singvals = torch.linalg.svdvals(mat)
    norm = torch.norm(singvals, p=1).sum()

    return norm

def schatten_proj_2d(hessian, im_size_x, im_size_y, device):

    a, b, c = hessian[: ,0], hessian[: ,2], hessian[:,1]
    mat = torch.cat((a, b, b, c), dim=0).view(2, 2, -1).permute(2, 0, 1)
    u, singvals, v = torch.svd(mat)
    s_proj = torch.clip(singvals, -1, 1)
    hessian_rec = torch.matmul(torch.matmul(u, torch.diag_embed(s_proj)), v.mT)
    hess2 = hessian_rec.permute(1, 2, 0).view(4, hessian.size(2), hessian.size(3))
    hessian_rebuild = torch.cat((hess2[0:1, :, :], hess2[3:4, :, :], hess2[2:3, :, :]), dim=0)[None, ]
    return hessian_rebuild

def hessian_2d(x, filters): 

    filters1 = filters[0]; filters2 = filters[1]; filters3 = filters[2]

    hessian1_valid = F.conv2d(x, filters1)
    bound1 = x[:, :, :, -2:-1] - x[:, :, :, -1:]
    hessian1 = torch.cat((hessian1_valid, bound1, bound1), dim=3)
    
    hessian2_valid = F.conv2d(x, filters2)
    bound2 = x[:, :, -2:-1, :] - x[:, :, -1:, :]
    hessian2 = torch.cat((hessian2_valid, bound2, bound2 ), dim=2)

    hessian3_valid = F.conv2d(x, filters3)
    hessian3 = F.pad(hessian3_valid, (0, 1), "constant", 0)
    hessian3 = F.pad(hessian3, (0, 0, 0, 1), "constant", 0)

    hessian = torch.cat((hessian1, hessian2 , hessian3), dim=1)
    return hessian


def hessian_adj_2d(y, filters): 

    filters1 = filters[0]; filters2 = filters[1]; filters3 = filters[2]

    hessian_adj1 = F.conv_transpose2d(y[:, 0:1, :, :-2], filters1)
    bound_adj1 = y[:, 0:1, :, -2:-1] + y[:, 0:1, :, -1:]
    hessian_adj1[:, :, :, -2:-1] += bound_adj1
    hessian_adj1[:, :, :, -1:] -= bound_adj1

    hessian_adj2 = F.conv_transpose2d(y[:, 1:2, :-2, :], filters2)
    bound_adj2 = y[:, 1:2, -2:-1, :] + y[:, 1:2, -1:, :]
    hessian_adj2[:, :, -2:-1, :] += bound_adj2
    hessian_adj2[:, :, -1:, :] -= bound_adj2

    hessian_adj3 = F.conv_transpose2d(y[:, 2:3, :-1, :-1], filters3)
    
    hessian_adj = hessian_adj1 + hessian_adj2 + hessian_adj3 

    return hessian_adj

def prox_svd2d(y, niter, lmbda, im_size_x, im_size_y, device, verbose=False):

    filters1 = torch.Tensor([[[[1., -2., 1]]]]).to(device).double()
    filters2 = torch.Tensor([[[[1], [-2], [1]]]]).to(device).double()
    filters3 = torch.Tensor([[[[1., -1.], [-1., 1]]]]).to(device).double()
    filters = [filters1, filters2, filters3]

    omg_k = torch.zeros((1, 3,im_size_x, im_size_y), requires_grad=False, device=device).double()
    psi_k = torch.zeros((1, 3,im_size_x, im_size_y), requires_grad=False, device=device).double()
    t_k = 1 

    alpha = 1/(64*lmbda)

    if verbose: 
        log_dir = 'ExpA/2d/loss/svd.txt'
        loss_list = list()

    for _ in range(niter):
            pc = torch.clip(y - lmbda * hessian_adj_2d(psi_k, filters), 0, 1)
            Lpc = hessian_2d(pc, filters)
            omg_kp1 = schatten_proj_2d(psi_k + alpha * Lpc, im_size_x, im_size_y, device)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            psi_kp1 = omg_k + (t_k - 1) / t_kp1 * (omg_kp1 - omg_k)                                                                                                                                                                                                                                                                                                                                         

            if verbose: 
                loss = ((y - pc)**2).sum() / 2 + lmbda * schatten_norm_2d(Lpc, im_size_x, im_size_y, device)
                loss_list.append(loss.item())

            omg_k = omg_kp1
            psi_k = psi_kp1
            t_k = t_kp1

    if verbose: 
        loss_array = np.array(loss_list)
        np.savetxt(log_dir, loss_array)

    c = torch.clip(y - lmbda * hessian_adj_2d(omg_k, filters), 0, 1)

    return c 


# --------------------------------------------- 2d prox htv svd (closed form)  -----------------------------------------------

def schatten_norm_2d_closed(hessian, im_size_x, im_size_y, device):

    ii, ij, ji, jj = (0, 2, 2, 1)

    tr = hessian[0, ii, ...] + hessian[0, jj, ...]
    dt = (hessian[0, ii, ...] - hessian[0, jj, ...])**2 + 4*hessian[0, ij, ...]*hessian[0, ji, ...]

    singvals = torch.zeros([2, 1, im_size_x, im_size_y]).to(device).double()
    singvals[0, ...] = 0.5*(tr+torch.sqrt(dt))
    singvals[1, ...] = 0.5*(tr-torch.sqrt(dt))

    norm = torch.norm(singvals, p=1).sum()

    return norm

def schatten_proj_2d_closed(hessian, im_size_x, im_size_y, device):

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

def hessian_2d_closed(x, filters): 
    
    filters1 = filters[0]; filters2 = filters[1]; filters3 = filters[2]

    hessian1_valid = F.conv2d(x, filters1)
    bound1 = x[:, :, :, -2:-1] - x[:, :, :, -1:]
    hessian1 = torch.cat((hessian1_valid, bound1, bound1), dim=3)
    
    hessian2_valid = F.conv2d(x, filters2)
    bound2 = x[:, :, -2:-1, :] - x[:, :, -1:, :]
    hessian2 = torch.cat((hessian2_valid, bound2, bound2 ), dim=2)

    hessian3_valid = F.conv2d(x, filters3)
    hessian3 = F.pad(hessian3_valid, (0, 1), "constant", 0)
    hessian3 = F.pad(hessian3, (0, 0, 0, 1), "constant", 0)


    hessian = torch.cat((hessian1, hessian2 , hessian3), dim=1)
    return hessian


def hessian_adj_2d_closed(y, filters): 

    filters1 = filters[0]; filters2 = filters[1]; filters3 = filters[2]

    hessian_adj1 = F.conv_transpose2d(y[:, 0:1, :, :-2], filters1)
    bound_adj1 = y[:, 0:1, :, -2:-1] + y[:, 0:1, :, -1:]
    hessian_adj1[:, :, :, -2:-1] += bound_adj1
    hessian_adj1[:, :, :, -1:] -= bound_adj1

    hessian_adj2 = F.conv_transpose2d(y[:, 1:2, :-2, :], filters2)
    bound_adj2 = y[:, 1:2, -2:-1, :] + y[:, 1:2, -1:, :]
    hessian_adj2[:, :, -2:-1, :] += bound_adj2
    hessian_adj2[:, :, -1:, :] -= bound_adj2

    hessian_adj3 = F.conv_transpose2d(y[:, 2:3, :-1, :-1], filters3)
    
    hessian_adj = hessian_adj1 + hessian_adj2 + hessian_adj3 

    return hessian_adj

def prox_svd2d_closed(y, niter, lmbda, im_size_x, im_size_y, device, verbose=False):

    filters1 = torch.Tensor([[[[1., -2., 1]]]]).to(device).double()
    filters2 = torch.Tensor([[[[1], [-2], [1]]]]).to(device).double()
    filters3 = torch.Tensor([[[[1., -1.], [-1., 1]]]]).to(device).double()
    filters = [filters1, filters2, filters3]

    omg_k = torch.zeros((1, 3,im_size_x, im_size_y), requires_grad=False, device=device).double()
    psi_k = torch.zeros((1, 3,im_size_x, im_size_y), requires_grad=False, device=device).double()
    t_k = 1 

    alpha = 1/(64*lmbda)

    if verbose: 
        log_dir = 'ExpA/2d/loss/svd_closed.txt'
        loss_list = list()

    for _ in range(niter):
            pc = torch.clip(y - lmbda * hessian_adj_2d_closed(psi_k, filters), 0, 1)
            Lpc = hessian_2d_closed(pc, filters)
            omg_kp1 = schatten_proj_2d_closed(psi_k + alpha * Lpc, im_size_x, im_size_y, device)
            t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
            psi_kp1 = omg_k + (t_k - 1) / t_kp1 * (omg_kp1 - omg_k)                                                                                                                                                                                                                                                                                                                                         

            if verbose: 
                loss = ((y - pc)**2).sum() / 2 + lmbda * schatten_norm_2d_closed(Lpc, im_size_x, im_size_y, device)
                loss_list.append(loss.item())

            omg_k = omg_kp1
            psi_k = psi_kp1
            t_k = t_kp1

    if verbose: 
        loss_array = np.array(loss_list)
        np.savetxt(log_dir, loss_array)

    c = torch.clip(y - lmbda * hessian_adj_2d_closed(omg_k, filters), 0, 1)

    return c 

# --------------------------------------------- 2d prox htv box -----------------------------------------------

def L_box_htv2d(x, filters): 
    filters1 = filters[0]; filters2 = filters[1]; filters3 = filters[2]
    
    Lx1 = F.pad(F.conv2d(x, filters1), (0, 1, 0, 1), "constant", 0)
    Lx2 = F.pad(F.conv2d(x, filters2), (0, 1, 0, 2), "constant", 0)
    Lx3 = F.pad(F.conv2d(x, filters3), (0, 2, 0, 1), "constant", 0)

    Lx = torch.cat((Lx1, Lx2, Lx3), dim=1)

    return Lx 

def Lt_box_htv2d(y, filters): 
    filters1 = filters[0]; filters2 = filters[1]; filters3 = filters[2]
    
    Lt1y = F.conv_transpose2d(y[:, 0, :-1, :-1], filters1)
    Lt2y = F.conv_transpose2d(y[:, 1, :-2, :-1], filters2)
    Lt3y = F.conv_transpose2d(y[:, 2, :-1, :-2], filters3)

    Lty = Lt1y + Lt2y + Lt3y

    return Lty

def prox_box_htv2d(y, niter, lmbda, im_size_x, im_size_y, device, verbose=False):

    filters1 = 2 * torch.Tensor([[[[1., -1.,], [-1., 1.]]]]).to(device).double()
    filters2 = torch.Tensor([[[[1., 0.], [-1., -1.], [0., 1.]]]]).to(device).double()
    filters3 = torch.Tensor([[[[1., -1., 0.], [0., -1., 1.]]]]).to(device).double()
    filters = [filters1, filters2, filters3]

    v_k = torch.zeros((1, 3,im_size_x, im_size_y), requires_grad=False, device=device).double()
    u_k = torch.zeros((1, 3,im_size_x, im_size_y), requires_grad=False, device=device).double()

    t_k = 1 

    alpha = 1 / 64
    
    if verbose: 
        log_dir = 'ExpA/2d/loss/box.txt'
        loss_list = list()

    for _ in range(niter):
        LTv = Lt_box_htv2d(v_k, filters)
        pc = torch.clip(y - LTv, 0, 1)
        Lpc = L_box_htv2d(pc, filters)

        if verbose: 
            loss = ((y - pc)**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=1).sum()
            loss_list.append(loss.item())

        u_kp1 = torch.clip(v_k + alpha * Lpc, -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

        u_k = u_kp1
        v_k = v_kp1
        t_k = t_kp1

    c = y - Lt_box_htv2d(u_k, filters)
    c = torch.clip(c, 0, 1)

    if verbose: 
        loss_array = np.array(loss_list)
        np.savetxt(log_dir, loss_array)

    return c 

# --------------------------------------------- 3d prox htv box -----------------------------------------------


def prox_box3d(y, niter, lmbda, im_size_x, im_size_y, im_size_z, device, verbose=False):

    filters = torch.zeros([6, 1, 3, 3, 3]).to(device).double()
    filters[0, 0, 0, 0, 0] = 2.; filters[0, 0, 1, 0, 0] = -2.; filters[0, 0, 0, 1, 0] = -2.; filters[0, 0, 1, 1, 0] = 2.
    filters[1, 0, 0, 0, 0] = 2.; filters[1, 0, 0, 1, 0] = -2.; filters[1, 0, 0, 0, 1] = -2.; filters[1, 0, 0, 1, 1] = 2.
    filters[2, 0, 0, 0, 0] = 2.; filters[2, 0, 1, 0, 0] = -2.; filters[2, 0, 0, 0, 1] = -2.; filters[2, 0, 1, 0, 1] = 2.
    filters[3, 0, 0, 0, 0] = 1.; filters[3, 0, 1, 0, 0] = -1.; filters[3, 0, 1, 1, 1] = -1.; filters[3, 0, 2, 1, 1] = 1.
    filters[4, 0, 0, 0, 0] = 1.; filters[4, 0, 0, 1, 0] = -1.; filters[4, 0, 1, 1, 1] = -1.; filters[4, 0, 1, 2, 1] = 1.
    filters[5, 0, 0, 0, 0] = 1.; filters[5, 0, 0, 0, 1] = -1.; filters[5, 0, 1, 1, 1] = -1.; filters[5, 0, 1, 1, 2] = 1.


    v_k = torch.nn.functional.conv3d(y, filters, padding=1)
    u_k = torch.nn.functional.conv3d(y, filters, padding=1)

    t_k = 1 

    alpha = 1 / 240
    
    if verbose: 
        log_dir = 'ExpA/3d/loss/box.txt'
        loss_list = list()

    for _ in range(niter):
        LTv = torch.nn.functional.conv_transpose3d(v_k, filters, padding=1)
        pc = torch.clip(y - LTv, 0, 1)
        Lpc = torch.nn.functional.conv3d(pc, filters, padding=1)

        if verbose: 
            loss = ((y - pc)**2).sum() / 2 + lmbda * torch.norm(Lpc, dim=1, p=1).sum()
            loss_list.append(loss.item())

        u_kp1 = torch.clip(v_k + alpha * Lpc, -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

        u_k = u_kp1
        v_k = v_kp1
        t_k = t_kp1

    c = y - torch.nn.functional.conv_transpose3d(u_k, filters, padding=1)
    c = torch.clip(c, 0, 1)

    if verbose: 
        loss_array = np.array(loss_list)
        np.savetxt(log_dir, loss_array)

    return c 


# --------------------------------------------- 3d prox htv svd closed -----------------------------------------------

def schatten_norm_3d_closed(hessian, device):
    
    shape_out = (hessian.shape[0], *hessian.shape[2:], 3)
        
    a, b, c, d, e, f = hessian[: ,0], hessian[: ,2], hessian[:,5], hessian[:,1], hessian[:,4], hessian[:,3]

    
    tr = a + b + c
    y = a*a + b*b + c*c - a*b -a*c -b*c + 3*(d*d + f*f + e*e)
    z = -(2*a-b-c)*(2*b-a-c)*(2*c-a-b) \
            + 9*( (2*a-b-c)*e*e + (2*b-a-c)*f*f + (2*c-a-b)*d*d  ) \
            - 54*d*e*f
    

    sqy = torch.sqrt(y)
    phi = (torch.pi/2)*torch.ones_like(a)

    
    phi[z!=0] = torch.arctan(torch.sqrt(torch.clip(4*y[z!=0]**3 - z[z!=0]**2, 0, float('inf')))/z[z!=0])
    phi[z<0] += torch.pi

    singvals = torch.zeros(shape_out).to(device).double()
    singvals[..., 1] = (tr - 2*sqy*torch.cos(phi/3) )/3
    singvals[..., 0] = (tr + 2*sqy*torch.cos((phi-torch.pi)/3) )/3
    singvals[..., 2] = (tr + 2*sqy*torch.cos((phi+torch.pi)/3) )/3
    
    norm = torch.norm(singvals, p=1).sum()

    return norm

    
def schatten_proj_3d_closed(hessian, device, tol=1e-12):
    
    shape_out = (hessian.shape[0], *hessian.shape[2:], 3) # hermit output shape

    a, b, c, d, e, f = hessian[: ,0], hessian[: ,2], hessian[:,5], hessian[:,1], hessian[:,4], hessian[:,3]

    tr = a + b + c
    y = a*a + b*b + c*c - a*b -a*c -b*c + 3*(d*d + f*f + e*e)
    z = -(2*a-b-c)*(2*b-a-c)*(2*c-a-b) \
            + 9*( (2*a-b-c)*e*e + (2*b-a-c)*f*f + (2*c-a-b)*d*d  ) \
            - 54*d*e*f
    

    sqy = torch.sqrt(y)
    phi = (torch.pi/2)*torch.ones_like(a)

    phi[z!=0] = torch.arctan(torch.sqrt(torch.clip(4*y[z!=0]**3 - z[z!=0]**2, 0, float('inf')))/z[z!=0])
    phi[z<0] += torch.pi

    
    singvals = torch.zeros(shape_out).to(device).double()
    singvals[..., 1] = (tr - 2*sqy*torch.cos(phi/3) )/3
    singvals[..., 2] = (tr + 2*sqy*torch.cos((phi-torch.pi)/3) )/3
    singvals[..., 0] = (tr + 2*sqy*torch.cos((phi+torch.pi)/3) )/3

    singvecs = torch.zeros(*shape_out, 3).to(device).double()
    
    fis0  = torch.abs(f)<tol
    
    for i in range(3): # for the three eigenvectors
        
        m = torch.where(fis0,
                            (f*f - (c-singvals[..., i])*(a-singvals[..., i]))/(e*(a-singvals[..., i])-d*f),
                            (d*(c-singvals[..., i])-e*f)/(f*(b-singvals[..., i])-d*e)
                            )
    
        singvecs[..., i, 2] = 1
        
        singvecs[..., i, 1] = m
        
        singvecs[..., i, 0] = torch.where(fis0,  (m*(singvals[..., i]-b) - e)/d, (singvals[..., i] - c - e*m)/f)        
         

    dis0 = torch.abs(d)<tol; eis0 = torch.abs(e)<tol
    def0 = dis0 & fis0 & eis0; defnot0 = ~def0
    df0 = dis0 & fis0 & defnot0
    de0 = dis0 & eis0 & defnot0
    ef0 = eis0 & fis0 & defnot0
    
    singvecs[df0 | de0 | ef0 | def0, ...] = 0

    singvecs[..., 0, 0][def0] = singvecs[..., 1, 1][def0] = singvecs[..., 2, 2][def0] = 1.

    singvecs[..., 0, 0][df0] =  singvecs[..., 1, 1][de0] =  singvecs[..., 2, 2][ef0] = 1.

    singvecs[..., 1, 1][df0] = -e[df0]
    singvecs[..., 1, 2][df0] = b[df0]-singvals[..., 1][df0]
    singvecs[..., 2, 1][df0] = -singvecs[..., 1, 2][df0]
    singvecs[..., 2, 2][df0] = singvecs[..., 1, 1][df0]

    singvecs[..., 0, 0][de0] = -f[de0]
    singvecs[..., 0, 2][de0] = a[de0]-singvals[..., 1][de0]
    singvecs[..., 2, 0][de0] = -singvecs[..., 0, 2][de0]
    singvecs[..., 2, 2][de0] = singvecs[..., 0, 0][de0]

    singvals[de0] = torch.flip(torch.roll(singvals[de0],1), [0])

    singvecs[..., 1, 0][ef0] = -d[ef0]
    singvecs[..., 1, 1][ef0] = a[ef0]-singvals[..., 1][ef0]
    singvecs[..., 0, 0][ef0] = -singvecs[..., 1, 1][ef0]
    singvecs[..., 0, 1][ef0] = singvecs[..., 1, 0][ef0]
    
    
    nd = len(shape_out)
    singvecs = (singvecs /torch.norm(singvecs, p=2, dim=nd)[..., None] ) #.transpose(nd, nd-1)

    # project 
    singvals_prj = torch.clip(singvals, -1, 1)
    s = singvals_prj
    V = singvecs

    hessian_rebuild = torch.zeros((hessian.shape)).to(device).double()
    # rebuild
    hessian_rebuild[:, 0] = s[..., 0]*V[..., 0, 0]**2 + s[..., 1]*V[..., 1, 0]**2 + s[..., 2]*V[..., 2, 0]**2
    hessian_rebuild[:, 2] = s[..., 0]*V[..., 0, 1]**2 + s[..., 1]*V[..., 1, 1]**2 + s[..., 2]*V[..., 2, 1]**2
    hessian_rebuild[:, 5] = s[..., 0]*V[..., 0, 2]**2 + s[..., 1]*V[..., 1, 2]**2 + s[..., 2]*V[..., 2, 2]**2
    hessian_rebuild[:, 1] = s[..., 0]*V[..., 0, 0]*V[..., 0, 1] + s[..., 1]*V[..., 1, 0]*V[..., 1, 1] + s[..., 2]*V[..., 2, 0]*V[..., 2, 1]
    hessian_rebuild[:, 4] = s[..., 0]*V[..., 0, 2]*V[..., 0, 1] + s[..., 1]*V[..., 1, 2]*V[..., 1, 1] + s[..., 2]*V[..., 2, 2]*V[..., 2, 1]
    hessian_rebuild[:, 3] = s[..., 0]*V[..., 0, 0]*V[..., 0, 2] + s[..., 1]*V[..., 1, 0]*V[..., 1, 2] + s[..., 2]*V[..., 2, 0]*V[..., 2, 2]


    return hessian_rebuild




def hessian3_closed(x, filters): 

    hessian = torch.nn.functional.conv3d(x, filters, padding=1)
    return hessian

def hessian_adj3_closed(x, filters): 

    hessian_adj = torch.nn.functional.conv_transpose3d(x, filters, padding=1)
    return hessian_adj


def prox_svd3d_closed(y, niter, lmbda, im_size_x, im_size_y, im_size_z, device, verbose=False):

    filters = torch.zeros([6, 1, 3, 3, 3]).to(device).double()
    filters[0, 0, 0, 0, 0] = 1.; filters[0, 0, 0, 0, 1] = -2.; filters[0, 0, 0, 0, 2] = 1.
    filters[1, 0, 0, 0, 0] = 1.; filters[1, 0, 0, 1, 0] = -1.; filters[1, 0, 0, 0, 1] = -1.; filters[1, 0, 0, 1, 1] = 1.
    filters[2, 0, 0, 0, 0] = 1.; filters[2, 0, 0, 1, 0] = -2.; filters[2, 0, 0, 2, 0] = 1.; 
    filters[3, 0, 0, 0, 0] = 1.; filters[3, 0, 0, 0, 1] = -1.; filters[3, 0, 1, 0, 0] = -1.; filters[3, 0, 1, 0, 1] = 1.
    filters[4, 0, 0, 0, 0] = 1.; filters[4, 0, 0, 1, 0] = -1.; filters[4, 0, 1, 0, 0] = -1.; filters[4, 0, 1, 1, 0] = 1.
    filters[5, 0, 0, 0, 0] = 1.; filters[5, 0, 1, 0, 0] = -2.; filters[5, 0, 2, 0, 0] = 1.; 

    omg_k = hessian3_closed(y, filters)
    psi_k = hessian3_closed(y, filters)
    t_k = 1 

    alpha = 1/(144*lmbda)

    if verbose: 
        log_dir = 'ExpA/3d/loss/svd_closed.txt'
        loss_list = list()

    for _ in range(niter):
        pc = torch.clip(y - lmbda * hessian_adj3_closed(psi_k, filters), 0, 1)
        Lpc = hessian3_closed(pc, filters)
        omg_kp1 = schatten_proj_3d_closed(psi_k + alpha * Lpc, device)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        psi_kp1 = omg_k + (t_k - 1) / t_kp1 * (omg_kp1 - omg_k)                                                                                                                                                                                                                                                                                                                                         

        if verbose: 
            loss = ((y - pc)**2).sum() / 2 + lmbda * schatten_norm_3d_closed(Lpc, device)
            loss_list.append(loss.item())

        omg_k = omg_kp1
        psi_k = psi_kp1
        t_k = t_kp1

    if verbose: 
        loss_array = np.array(loss_list)
        np.savetxt(log_dir, loss_array)

    c = torch.clip(y - lmbda * hessian_adj3_closed(omg_k, filters), 0, 1)

    return c 


# --------------------------------------------- 3d prox htv svd -----------------------------------------------

def schatten_norm_3d(hessian, device):
    
    a, b, c, d, e, f = hessian[: ,0], hessian[: ,2], hessian[:,5], hessian[:,1], hessian[:,3], hessian[:,4]
    mat = torch.cat((a, d, e, d, b, f, e, f, c), dim=0).view(3, 3, -1).permute(2, 0, 1)
    singvals = torch.linalg.svdvals(mat)
    norm = torch.norm(singvals, p=1).sum()
    return norm

    
def schatten_proj_3d(hessian, device, tol=1e-3):

    a, b, c, d, e, f = hessian[: ,0], hessian[: ,2], hessian[:,5], hessian[:,1], hessian[:,3], hessian[:,4]
    mat = torch.cat((a, d, e, d, b, f, e, f, c), dim=0).view(3, 3, -1).permute(2, 0, 1)
    u, singvals, v = torch.svd(mat)
    s_proj = torch.clip(singvals, -1, 1)
    hessian_rec = torch.matmul(torch.matmul(u, torch.diag_embed(s_proj)), v.mT)
    hess2 = hessian_rec.permute(1, 2, 0).view(9, hessian.size(2), hessian.size(3), hessian.size(4))
    hessian_rebuild = torch.cat((hess2[0:1, :, :, :], hess2[3:4, :, :, :], hess2[4:5, :, :, :], hess2[2:3, :, :, :], hess2[5:6, :, :, :], hess2[8:9, :, :, :]), dim=0)[None, ]
    return hessian_rebuild

def hessian3(x, filters): 

    hessian = torch.nn.functional.conv3d(x, filters, padding=1)
    return hessian

def hessian_adj3(x, filters): 

    hessian_adj = torch.nn.functional.conv_transpose3d(x, filters, padding=1)
    return hessian_adj

def prox_svd3d(y, niter, lmbda, im_size_x, im_size_y, im_size_z, device, verbose=False):
    
    filters = torch.zeros([6, 1, 3, 3, 3]).to(device).double()
    filters[0, 0, 0, 0, 0] = 1.; filters[0, 0, 0, 0, 1] = -2.; filters[0, 0, 0, 0, 2] = 1.
    filters[1, 0, 0, 0, 0] = 1.; filters[1, 0, 0, 1, 0] = -1.; filters[1, 0, 0, 0, 1] = -1.; filters[1, 0, 0, 1, 1] = 1.
    filters[2, 0, 0, 0, 0] = 1.; filters[2, 0, 0, 1, 0] = -2.; filters[2, 0, 0, 2, 0] = 1.; 
    filters[3, 0, 0, 0, 0] = 1.; filters[3, 0, 0, 0, 1] = -1.; filters[3, 0, 1, 0, 0] = -1.; filters[3, 0, 1, 0, 1] = 1.
    filters[4, 0, 0, 0, 0] = 1.; filters[4, 0, 0, 1, 0] = -1.; filters[4, 0, 1, 0, 0] = -1.; filters[4, 0, 1, 1, 0] = 1.
    filters[5, 0, 0, 0, 0] = 1.; filters[5, 0, 1, 0, 0] = -2.; filters[5, 0, 2, 0, 0] = 1.; 

    omg_k = hessian3(y, filters)
    psi_k = hessian3(y, filters)
    t_k = 1
    alpha = 1/(144*lmbda)

    if verbose: 
        log_dir = 'ExpA/3d/loss/svd.txt'
        loss_list = list()

    for _ in range(niter):
        pc = torch.clip(y - lmbda * hessian_adj3(psi_k, filters), 0, 1)
        Lpc = hessian3(pc, filters)
        omg_kp1 = schatten_proj_3d(psi_k + alpha * Lpc, device)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        psi_kp1 = omg_k + (t_k - 1) / t_kp1 * (omg_kp1 - omg_k)                                                                                                                                                                                                                                                                                                                                         

        if verbose: 
            
            loss = ((y - pc)**2).sum() / 2 + lmbda * schatten_norm_3d(Lpc, device)
            loss_list.append(loss.item())

        omg_k = omg_kp1
        psi_k = psi_kp1
        t_k = t_kp1

    if verbose: 
        loss_array = np.array(loss_list)
        np.savetxt(log_dir, loss_array)

    c = torch.clip(y - lmbda * hessian_adj3(omg_k, filters), 0, 1)

    return c 



if __name__ == __name__: 
    pass