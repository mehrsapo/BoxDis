from __init__ import * 


device = 'cpu'
sigma = 25

with open('ExpB/data/mri40frames.pickle', 'rb') as handle:
    img_numpy  = pickle.load(handle)

np.random.seed(2023)
(im_size_x, im_size_y, im_size_z) = img_numpy.shape
img_numpy = (img_numpy - np.min(np.min(img_numpy))) / np.max(np.max(img_numpy))
img_noisy = img_numpy  +  sigma / 255 * np.random.normal(0, 1, size=img_numpy.shape)

img_tensor = torch.from_numpy(img_numpy).float().view(1, 1, *img_numpy.shape).to(device)
y =  torch.from_numpy(img_noisy).float().view(1, 1, *img_numpy.shape).to(device).double()


t0 = time.time()
for _ in range(1):
    c_svd_open = prox_svd3d(y, 200, 0.02, img_tensor.size(2), img_tensor.size(3), img_tensor.size(4), device, verbose=False)
t1 = time.time()

if device != 'cpu':
    c_svd = prox_svd3d(y, 200, 0.02, img_tensor.size(2), img_tensor.size(3), img_tensor.size(4), device, verbose=True)
    torch.save(c_svd.cpu(), 'ExpB/3d/signal/svd.pt')

if device != 'cpu':
    device = 'gpu'

np.savetxt('ExpB/3d/time/svd_' + device + '.txt', np.array([(t1-t0)/1]))