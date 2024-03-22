from __init__ import *

y = torch.load('ExpB/part1/y.pt')

N1 = y.size(2)

n_in = 20000
lmbda = 0.5

device = 'cpu'
stop = 1e-6

if stop == -1:
    stop_str = ''
else:
    stop_str = str(stop)

y = y.double().to(device)

for prox_gen in [prox_htv_inexact, prox_htv_svd]:
    prox = prox_gen(device, box_const=[0, 1])
    t0 = time.time()
    sol = prox.eval(y, N1, n_in, lmbda, verbose=True, stop=stop)
    t1 = time.time()
    timed = t1 - t0
    loss = prox.loss_list

    if device != 'cpu':
        device_n = 'gpu'
    else:
        device_n = 'cpu'

    torch.save(sol, 'ExpB/part1/sol_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')
    torch.save(loss, 'ExpB/part1/loss_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')
    torch.save(timed, 'ExpB/part1/time_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')

    if stop > 0:
        torch.save(prox.niters, 'ExpB/part1/iters_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')

    del prox, t0, t1, timed, sol, loss


for prox_gen in [prox_tv_iso, prox_tv_l2_inexact, prox_tv_upwind]:
    prox = prox_gen(device, box_const=[0, 1])
    t0 = time.time()
    sol = prox.eval(y, N1, n_in, lmbda, 1, verbose=True, stop=stop)
    t1 = time.time()
    timed = t1 - t0
    loss = prox.loss_list

    if device != 'cpu':
        device_n = 'gpu'
    else:
        device_n = 'cpu'

    torch.save(sol, 'ExpB/part1/sol_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')
    torch.save(loss, 'ExpB/part1/loss_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')
    torch.save(timed, 'ExpB/part1/time_' + prox_gen.__name__ + '_' + device_n + stop_str +'.pt')

    if stop > 0:
        torch.save(prox.niters, 'ExpB/part1/iters_' + prox_gen.__name__ + '_' + device_n + stop_str + '.pt')

    del prox, t0, t1, timed, sol, loss