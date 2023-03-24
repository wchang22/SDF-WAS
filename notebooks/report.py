#%%

import os
import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

mi.set_variant('cuda_ad_rgb')

from mitsuba import ScalarTransform4f as T

def relse(a, b):
    return dr.sqr(a - b) / (dr.sqr(b) + 1e-2)

def relmse(a, b):
    return dr.mean(relse(a, b))

def convert_to_lum(grad_tensor):
    grad_color = dr.unravel(mi.Color3f, dr.ravel(grad_tensor[...,:3]))
    grad_lum = mi.luminance(grad_color)
    return mi.TensorXf(grad_lum, shape=(grad_tensor.shape[0], grad_tensor.shape[1]))

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

#%% Define scene
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'sdf_integrator',
    },
    'sensor': {
        'type': 'perspective',
        'fov': 39.3077,
        'to_world': T.look_at(
            origin=[0, 0, 5],
            target=[0, 0, 0],
            up=[0, 1, 0]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 16
        },
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
            'rfilter': {
                'type': 'gaussian',
            },
            'pixel_format': 'rgb',
            'sample_border': True,
        },
    },
    'sphere0': {
        'type': 'sphere',
        'to_world': T.translate([0, 0, 0]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0, 1, 0]},
        },
    }
})

params = mi.traverse(scene)

#%% Render initial state
image_ref = mi.render(scene, params, seed=0, spp=1)
plt.axis('off')
plt.imshow(mi.util.convert_to_bitmap(image_ref));

#%% Show forward autodiff comparison with finite differences
key = 'sphere0.to_world'

def apply_transformation(params, t):
    params[key] = mi.Transform4f.translate([t.x, t.y, t.z])
    params.update()

def compute_fd(trans, eps=1e-3, spp=256):
    fd = None
    for c in range(3):
        trans_eps = mi.Vector3f(0)
        trans_eps[c] = eps
    
        apply_transformation(params, trans-trans_eps)
        image1 = mi.render(scene, params, seed=0, spp=spp)

        apply_transformation(params, trans+trans_eps)
        image2 = mi.render(scene, params, seed=0, spp=spp)

        if fd is None:
            fd = dr.zeros(mi.TensorXf, shape=image1.shape)
        fd += (image2 - image1) / (2 * eps)
    return fd

scene.integrator().reparam_𝜆d = 1e2
scene.integrator().reparam_exp = 10

trans = mi.Vector3f(0.5, -0.2, 0.3)
dr.enable_grad(trans)
apply_transformation(params, trans)

image = mi.render(scene, params, seed=0, spp=16)

dr.forward(trans)

# Fetch the image gradient values
grad_image = dr.grad(image)
grad_image_lum = convert_to_lum(grad_image)
fd_image = compute_fd(trans)
fd_image_lum = convert_to_lum(fd_image)
err_image = relse(grad_image, fd_image)
err_image_lum = convert_to_lum(err_image)

cmap = cm.coolwarm
vlim = np.quantile(dr.abs(grad_image_lum), 0.995)
err_vmax = np.quantile(err_image_lum, 0.999)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(grad_image_lum, cmap=cm.coolwarm, vmin=-vlim, vmax=vlim);
ax[0].set_title("Warped-Area Sampling");
ax[1].imshow(fd_image_lum, cmap=cm.coolwarm, vmin=-vlim, vmax=vlim);
ax[1].set_title("Finite Difference");
ax[2].imshow(err_image_lum, cmap=cm.viridis, vmin=0, vmax=err_vmax);
ax[2].set_title("relMSE");

#%% Optimization

opt = mi.ad.Adam(lr=0.01)
opt['trans'] = mi.Vector3f(0.6, -0.7, 0.7)

scene.integrator().reparam_𝜆d = 1e-1
scene.integrator().reparam_exp = 4

def apply_transformation(params, opt):
    key = 'sphere0.to_world'
    params[key] = mi.Transform4f.translate([opt['trans'].x, opt['trans'].y, opt['trans'].z])
    params.update()

apply_transformation(params, opt)
image_initial = mi.render(scene, params, seed=0, spp=16)

losses = []
n_iterations = 200
for i in range(n_iterations):
    apply_transformation(params, opt)

    image = mi.render(scene, params, seed=0, spp=1)

    loss = relmse(image, image_ref)
    dr.backward(loss)

    opt.step()

    losses.append(loss[0])
    print(f'Iteration {i} -- loss {loss[0]:.3e} --', end='\r')

    mi.util.write_bitmap(os.path.join(output_dir, f'{i}.png'), image);
print()

image = mi.render(scene, params, seed=0, spp=16)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
[axi.set_axis_off() for axi in ax.ravel()]
ax[0].imshow(mi.util.convert_to_bitmap(image_initial));
ax[0].set_title("Initial");
ax[1].imshow(mi.util.convert_to_bitmap(image));
ax[1].set_title("Optimized");
ax[2].imshow(mi.util.convert_to_bitmap(image_ref));
ax[2].set_title("Reference");

#%%
plt.plot(np.arange(n_iterations), losses);
plt.xlabel('Iteration');
plt.ylabel('Loss');
plt.title('Inverse rendering loss')
