from __init__ import * 

device = 'cuda:1'

y = torch.load('ExpA/data/y_PR.pt').to(device)
mask = torch.load('ExpA/data/mask_PR.pt').to(device)


t0 = time.time()
mrs = MultiResSolver_fista('fm', 'htv', lmbda =1e-5, mask=mask, h_init=1, range_r = 512, N_scales=1, device=device, verbose=True, toi=-1, box_const=None, obj_stop=True)
mrs.solve_fm(y)
np.savetxt('compare_cp/fista_inexact_loss.txt', np.array(mrs.losses[0]))
np.savetxt('compare_cp/fista_inexact_loss_iters.txt', np.array(mrs.iters))

t1 = time.time()
t = t1 - t0

np.savetxt('compare_cp/fista_inexact_times.txt', np.array([t]))