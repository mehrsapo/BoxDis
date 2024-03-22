from __init__ import * 

exp_name = 'ExpA_CS_pixel_tv'


device = 'cuda:3'

mask = torch.load('ExpA/data/mask_CS.pt').to(device)
y = torch.load('ExpA/data/y_CS.pt').to(device)

lmbda = 1e-6
toi = 1e-6 # -1
N_iter_in = 500
N_iter_out = 2000
N = 6
cond = 'pos' # 'none'

exp_name = exp_name + '_' + str(lmbda) + '_' + str(toi) + '_' +  str(N) + '_' + str(N_iter_in) + '_' + str(N_iter_out) + '_' + cond

print(exp_name)

if cond == 'pos':
    cons = [0, float('inf')]
elif cond == 'none':
    cons = None


mrs = MultiResSolverPixel('fm', 'tv_pixel', lmbda =lmbda, mask=mask, h_init=8, range_r = 512,
                        n_iters_out=[N_iter_out]*N, n_iters_in=[N_iter_in]*N, N_scales=N, device=device, verbose=True, toi=toi, box_const=cons)

t0 = time.time()
mrs.solve_fm(y)
t1 = time.time()

for i in range(N):
    torch.save(mrs.sols[i].to('cpu'), 'ExpA/saved_results/mrs' + str(i)+ '_' +exp_name +'.pt')
   
torch.save(mrs.costs, 'ExpA/saved_results/mrs_cost' + exp_name +'.pt' +'.pt')
torch.save(mrs.losses, 'ExpA/saved_results/mrs_loss' + exp_name +'.pt')
torch.save(mrs.reg_values, 'ExpA/saved_results/mrs_reg' + exp_name +'.pt')
torch.save(mrs.mses, 'ExpA/saved_results/mrs_mse' + exp_name +'.pt')
torch.save(mrs.iters, 'ExpA/saved_results/mrs_iters' + exp_name +'.pt')
torch.save(mrs.times, 'ExpA/saved_results/mrs_times' + exp_name +'.pt')

torch.save(t1-t0, 'ExpA/saved_results/mrs_time_total' + exp_name +'.pt')