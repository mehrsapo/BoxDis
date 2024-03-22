from __init__ import * 


device = 'cpu'
sigma = 25

torch.manual_seed(2023)

with open('ExpB/data/mri40frames.pickle', 'rb') as handle:
    img_numpy  = pickle.load(handle)

img = img_numpy[:, :, 20]
img_tensor = torch.from_numpy(img)[None, None, :, :].double().to(device) 
y = img_tensor + sigma / 255 * torch.normal(0, 1, size=img_tensor.size()).to(device)


t0 = time.time()
for _ in range(10):
    prox_svd2d_closed(y, 200, 0.03, img_tensor.size(2), img_tensor.size(3), device, verbose=False)
t1 = time.time()

if device != 'cpu':
    c_svd_closed = prox_svd2d_closed(y, 200, 0.03, img_tensor.size(2), img_tensor.size(3), device, verbose=True)
    torch.save(c_svd_closed.cpu(), 'ExpB/2d/signal/svd_closed.pt')

if device != 'cpu':
    device = 'gpu'

np.savetxt('ExpB/2d/time/svd_closed_' + device + '.txt', np.array([(t1-t0)/10]))