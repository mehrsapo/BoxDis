from __init__ import * 


device = 'cuda:3'
sigma = 25

with open('mri40frames.pickle', 'rb') as handle:
    img_numpy  = pickle.load(handle)

np.random.seed(2023)
(im_size_x, im_size_y, im_size_z) = img_numpy.shape
img_numpy = (img_numpy - np.min(np.min(img_numpy))) / np.max(np.max(img_numpy))
img_noisy = img_numpy  +  sigma/255 * np.random.normal(0, 1, size=img_numpy.shape)

img_tensor = torch.from_numpy(img_numpy).float().view(1, 1, *img_numpy.shape).to(device)
y =  torch.from_numpy(img_noisy).float().view(1, 1, *img_numpy.shape).to(device).double()


t0 = time.time()
for _ in range(10):
    prox_box3d(y, 200, 0.01, img_tensor.size(2), img_tensor.size(3), img_tensor.size(4), device, verbose=False)
t1 = time.time()

c_box = prox_box3d(y, 200, 0.01, img_tensor.size(2), img_tensor.size(3), img_tensor.size(4), device, verbose=True)

torch.save(c_box.cpu(), 'ExpA/3d/signal/box.pt')
np.savetxt('ExpA/3d/time/box.txt', np.array([(t1-t0)/10]))