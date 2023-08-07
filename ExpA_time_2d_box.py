from __init__ import * 


device = 'cuda:0'
sigma = 25

torch.manual_seed(2023)

with open('mri40frames.pickle', 'rb') as handle:
    img_numpy  = pickle.load(handle)

img = img_numpy[:, :, 20]
img_tensor = torch.from_numpy(img)[None, None, :, :].double().to(device) 
y = img_tensor + sigma / 255 * torch.normal(0, 1, size=img_tensor.size()).to(device)


t0 = time.time()
for _ in range(10):
    prox_box_htv2d(y, 200, 0.02, img_tensor.size(2), img_tensor.size(3), device, verbose=False)
t1 = time.time()

c_box = prox_box_htv2d(y, 200, 0.02, img_tensor.size(2), img_tensor.size(3), device, verbose=True)

torch.save(c_box.cpu(), 'ExpA/2d/signal/box.pt')
np.savetxt('ExpA/2d/time/box.txt', np.array([(t1-t0)/10]))