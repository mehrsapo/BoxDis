from __init__ import * 

device = 'cuda:0'

y = torch.load('ExpA/data/y_PR.pt').to(device)
mask = torch.load('ExpA/data/mask_PR.pt').to(device)

times = list()
taus = [1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 1]
for tau in taus:
    t0 = time.time()
    mrs_cp = MultiResSolver_cp('fm', 'htv', lmbda =1e-5, mask=mask, h_init=1, range_r = 512, N_scales=1, device=device, verbose=True, toi=-1, tau=tau, theta=1)
    mrs_cp.solve_fm(y)
    np.savetxt('compare_cp/cp_loss_theta1_tau'+str(tau)+'.txt', np.array(mrs_cp.losses[0]))
    t1 = time.time()
    t = t1 - t0
    times.append(t)

for tau in taus:
    t0 = time.time()
    mrs_cp = MultiResSolver_cp('fm', 'htv', lmbda =1e-5, mask=mask, h_init=1, range_r = 512, N_scales=1, device=device, verbose=True, toi=-1, tau=tau, theta=0.5)
    mrs_cp.solve_fm(y)
    np.savetxt('compare_cp/cp_loss_theta_half_tau'+str(tau)+'.txt', np.array(mrs_cp.losses[0]))
    t1 = time.time()
    t = t1 - t0
    times.append(t)

np.savetxt('compare_cp/cp_times.txt', np.array(times))