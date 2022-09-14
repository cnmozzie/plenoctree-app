import svox
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


#device = 'cuda:0'
device ='cpu'

t = svox.N3Tree.load("tree_opt.npz", device=device)
r = svox.VolumeRenderer(t)




class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

tree_spec_dict = {
    'data': r.tree._spec().data,
    'child': r.tree._spec().child,
    'parent_depth': r.tree._spec().parent_depth,
    'extra_data': r.tree._spec().extra_data,
    'offset': r.tree._spec().offset,
    'scaling': r.tree._spec().scaling,
    '_weight_accum': r.tree._spec()._weight_accum,
    '_weight_accum_max': r.tree._spec()._weight_accum_max
}

tree_spec_container = torch.jit.script(Container(tree_spec_dict))
tree_spec_container.save("tree_spec_dict.pt")

options_dict = {
    'step_size': r._get_options(False).step_size,
    'background_brightness': r._get_options(False).background_brightness,
    'format': r._get_options(False).format,
    'basis_dim': r._get_options(False).basis_dim,
    'ndc_width': r._get_options(False).ndc_width,
    'ndc_height': r._get_options(False).ndc_height,
    'ndc_focal': r._get_options(False).ndc_focal,
    'min_comp': r._get_options(False).min_comp,
    'max_comp': r._get_options(False).max_comp,
    'sigma_thresh': r._get_options(False).sigma_thresh,
    'stop_thresh': r._get_options(False).stop_thresh,
    'density_softplus': r._get_options(False).density_softplus,
    'rgb_padding': r._get_options(False).rgb_padding
}
options_container = torch.jit.script(Container(options_dict))
options_container.save("options_dict.pt")




'''
print(time.time())
for i in range(10):
    t1 = time.time()
    # Matrix copied from lego test set image 0
    c2w = pose_spherical(i*36,-30,4).to(device)

    with torch.no_grad():
        im = r.render_persp(c2w, height=800, width=800, fx=1111.111).clamp_(0.0, 1.0)
    t2 = time.time()
    print((t2-t1)*1000)
    #plt.imshow(im.cpu())
    #plt.savefig("test/lego_" + str(i) + ".png")
print(time.time())
'''
