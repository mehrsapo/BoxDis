from __init__ import * 

torch.manual_seed(2023)

device = 'cuda:0'
reg = 'htv'
task = 'fm'
image = 'mra'

if image == 'mra': 
    img = plt.imread('ExpB/data/mra.jpeg')[40:-25, 147:-206, 0] / 255
    img_tensor = torch.from_numpy(img)[None, None, :, :].double().to(device) 
    img_tensor = F.pad(torch.from_numpy(img)[None, None, :, :].double().to(device), (1, 1, 1, 1))


if task == 'fm': 
    mat = scipy.io.loadmat('ExpB/masks/radial_mask.mat')
    mask = torch.from_numpy(mat['mask']).double().to(device).view([1, 1, 256, 256])

    mri_in = img_tensor[:, :, :-1, :-1]
    mri = MRI(mask, device)
    mri.set_h(0.5)
    y = mri.H(mri_in)

def objective(trial):
    if task == 'fm':
        lmbda = trial.suggest_float('lmbda', 0, 20e-5)
        mrs = MultiResSolver('fm', reg, lmbda = lmbda, mask=mask, h_init=1, range_r = 512,  N_scales=1, device=device, verbose=False, toi=1e-4)
        mrs.solve_fm(y)
        c_hat = mrs.sols[0]

    psnr = compute_PSNR(c_hat, img_tensor, 1)

    return psnr

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
fig = optuna.visualization.plot_slice(study, params=["lmbda"])
trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
name_file = 'ExpB/reg_tuner/tuner_' + task + '_' + image + '_' + reg + 'toi_1e-4_range200u.png'
fig.update_layout(title=reg + ', best lambda is ' + str(study.best_params['lmbda']) + ', psnr is ' + str(trial_with_highest_accuracy.values[0]))
fig.write_image(name_file, format='png')