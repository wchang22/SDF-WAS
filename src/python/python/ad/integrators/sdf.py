from __future__ import annotations
from typing import Tuple # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi
import gc

from .common import ADIntegrator

def abs_grad(x):
    return dr.select(x >= 0, 1, -1)

def _evaluate_warp_field(params: mi.SceneParameters,
                         ray: mi.Ray3f,
                         t: mi.Float,
                         ray_march_eps: mi.Float,
                         𝜆d: mi.Float,
                         exponent: mi.Float,
                         active: mi.Bool):
    pos = ray.o + t * ray.d
    f, hit, dfdx, dfdxx, _ = eval_scene(params, pos, ray_march_eps, active)
    dfdx = dr.detach(dfdx)

    with dr.suspend_grad():
        sil_dot = dr.dot(dfdx, ray.d)
        S = dr.abs(f) + 𝜆d * dr.abs(sil_dot)
        w = dr.power(dr.rcp(S) + 1e-6, exponent)

        da_df = abs_grad(f)
        da_dsildot = abs_grad(sil_dot)

        ds_domega = da_df * t * dfdx + 𝜆d * da_dsildot * (dfdx + t * ray.d @ dfdxx)
        dw_domega = -exponent * dr.power(dr.rcp(S) + 1e-6, exponent+1) * ds_domega

    V_direct = -f * dfdx / dr.dot(dfdx, dfdx)

    return w, dw_domega, w * V_direct, dr.dot(dw_domega, V_direct), dr.detach(f), hit

class _ReparameterizeOp(dr.CustomOp):
    def eval(self, scene, params, ray, ray_march_max_it, ray_march_eps,
             𝜆d, exponent, active):
        # Stash all of this information for the forward/backward passes
        self.scene = scene
        self.params = params
        self.ray = dr.detach(ray)
        self.ray_march_max_it = ray_march_max_it
        self.ray_march_eps = ray_march_eps
        self.𝜆d = 𝜆d
        self.exponent = exponent
        self.active = active

        # The reparameterization is simply the identity in primal mode
        return self.ray.d, dr.full(mi.Float, 1, dr.width(ray))


    def forward(self):
        """
        Propagate the gradients in the forward direction to 'ray.d' and the
        jacobian determinant 'det'. From a warp field point of view, the
        derivative of 'ray.d' is the warp field direction at 'ray', and
        the derivative of 'det' is the divergence of the warp field at 'ray'.
        """

        # Initialize some accumulators
        Z = mi.Float(0.0)
        dZ = mi.Vector3f(0.0)
        grad_V = mi.Vector3f(0.0)
        grad_div_lhs = mi.Float(0.0)
        it = mi.UInt32(0)
        active = mi.Bool(self.active)
        t_total = mi.Float(0)

        ray_grad_o = self.grad_in('ray').o

        loop = mi.Loop(name="reparameterize_ray(): forward propagation",
                       state=lambda: (it, active, t_total, Z, dZ, grad_V, grad_div_lhs))

        # Unroll the entire loop in wavefront mode
        # loop.set_uniform(True) # TODO can we turn this back on? (see self.active in loop condition)
        loop.set_max_iterations(self.ray_march_max_it)
        loop.set_eval_stride(self.ray_march_max_it)

        while loop(active & (it < self.ray_march_max_it)):
            ray = mi.Ray3f(self.ray)
            dr.enable_grad(ray.o)
            dr.set_grad(ray.o, ray_grad_o)

            Z_i, dZ_i, V_i, div_lhs_i, t, hit = _evaluate_warp_field(
                self.params,
                ray,
                t_total,
                self.ray_march_eps,
                self.𝜆d,
                self.exponent,
                active)

            # Do not clear input vertex gradient
            dr.forward_to(V_i, div_lhs_i,
                          flags=dr.ADFlag.ClearEdges | dr.ADFlag.ClearInterior)

            Z += Z_i * t
            dZ += dZ_i * t
            grad_V += dr.grad(V_i) * t
            grad_div_lhs += dr.grad(div_lhs_i) * t

            t_total += t
            it += 1
            active &= ~hit

        inv_Z = dr.rcp(dr.maximum(Z, 1e-8))
        V_theta  = grad_V * inv_Z
        div_V_theta = (grad_div_lhs - dr.dot(V_theta, dZ)) * inv_Z

        # Ignore inactive lanes
        V_theta = dr.select(self.active, V_theta, 0.0)
        div_V_theta = dr.select(self.active, div_V_theta, 0.0)

        self.set_grad_out((V_theta, div_V_theta))

    def backward(self):
        grad_direction, grad_divergence = self.grad_out()

        # Ignore inactive lanes
        grad_direction  = dr.select(self.active, grad_direction, 0.0)
        grad_divergence = dr.select(self.active, grad_divergence, 0.0)

        with dr.suspend_grad():
            # We need to raymarch a first time to compute the
            # constants Z and dZ in order to properly weight the incoming gradients
            Z = mi.Float(0.0)
            dZ = mi.Vector3f(0.0)
            it = mi.UInt32(0)
            active = mi.Bool(self.active)
            t_total = mi.Float(0)

            loop = mi.Loop(name="reparameterize_ray(): weight normalization",
                           state=lambda: (it, active, t_total, Z, dZ))

            # Unroll the entire loop in wavefront mode
            # loop.set_uniform(True) # TODO can we turn this back on? (see self.active in loop condition)
            loop.set_max_iterations(self.ray_march_max_it)
            loop.set_eval_stride(self.ray_march_max_it)

            while loop(active & (it < self.ray_march_max_it)):
                Z_i, dZ_i, _, _, t, hit = _evaluate_warp_field(
                    self.params,
                    self.ray,
                    t_total,
                    self.ray_march_eps,
                    self.𝜆d,
                    self.exponent,
                    active)

                Z += Z_i * t
                dZ += dZ_i * t

                t_total += t
                it += 1
                active &= ~hit

        # Un-normalized values
        V = dr.zeros(mi.Vector3f, dr.width(Z))
        div_V_1 = dr.zeros(mi.Float, dr.width(Z))
        dr.enable_grad(V, div_V_1)

        # Compute normalized values
        Z = dr.maximum(Z, 1e-8)
        V_theta = V / Z
        divergence = (div_V_1 - dr.dot(V_theta, dZ)) / Z
        direction = dr.normalize(self.ray.d + V_theta)

        dr.set_grad(direction, grad_direction)
        dr.set_grad(divergence, grad_divergence)
        dr.enqueue(dr.ADMode.Backward, direction, divergence)
        dr.traverse(mi.Float, dr.ADMode.Backward)

        grad_V = dr.grad(V)
        grad_div_V_1 = dr.grad(div_V_1)

        it = mi.UInt32(0)
        ray_grad_o = mi.Point3f(0)
        active = mi.Bool(self.active)
        t_total = mi.Float(0)

        loop = mi.Loop(name="reparameterize_ray(): backpropagation",
                       state=lambda: (it, active, t_total))

        # Unroll the entire loop in wavefront mode
        # loop.set_uniform(True) # TODO can we turn this back on? (see self.active in loop condition)
        loop.set_max_iterations(self.ray_march_max_it)
        loop.set_eval_stride(self.ray_march_max_it)

        while loop(active & (it < self.ray_march_max_it)):
            ray = mi.Ray3f(self.ray)
            dr.enable_grad(ray.o)

            _, _, V_i, div_V_1_i, t, hit = _evaluate_warp_field(
                self.params,
                ray,
                t_total,
                self.ray_march_eps,
                self.𝜆d,
                self.exponent,
                active)
            V_i *= t
            div_V_1_i *= t

            dr.set_grad(V_i, grad_V)
            dr.set_grad(div_V_1_i, grad_div_V_1)
            dr.enqueue(dr.ADMode.Backward, V_i, div_V_1_i)
            dr.traverse(mi.Float, dr.ADMode.Backward, dr.ADFlag.ClearVertices)
            ray_grad_o += dr.grad(ray.o)
            t_total += t
            it += 1
            active &= ~hit

        ray_grad = dr.detach(dr.zeros(type(self.ray)))
        ray_grad.o = ray_grad_o
        self.set_grad_in('ray', ray_grad)

    def name(self):
        return "reparameterize_ray()"


def reparameterize_ray(scene: mi.Scene,
                        params: mi.SceneParameters,
                        ray: mi.Ray3f,
                        ray_march_max_it: int=32,
                        ray_march_eps=1e-3,
                        𝜆d: float=1e-1,
                        exponent: float=4.0,
                        active: mi.Bool = True
) -> Tuple[mi.Vector3f, mi.Float]:
    return dr.custom(_ReparameterizeOp, scene, params, ray,
                     ray_march_max_it, ray_march_eps, 𝜆d, exponent, active)


class _ReparamWrapper:
    # ReparamWrapper instances can be provided as dr.Loop state
    # variables. For this to work we must declare relevant fields
    DRJIT_STRUCT = { }

    def __init__(self,
                 scene : mi.Scene,
                 params: Any,
                 reparam: Callable[
                     [mi.Scene, mi.SceneParameters, mi.Vector2f,
                      float, float, mi.Bool],
                     Tuple[mi.Vector2f, mi.Float]]):

        self.scene = scene
        self.params = params
        self.reparam = reparam

        # Only link the reparameterization CustomOp to differentiable scene
        # parameters with the AD computation graph if they control shape
        # information (vertex positions, etc.)
        if isinstance(params, mi.SceneParameters):
            params = params.copy()
            params.keep(
                [
                    k for k in params.keys() \
                        if (params.flags(k) & mi.ParamFlags.Discontinuous) != 0
                ]
            )

    def __call__(self,
                 ray: mi.Ray3f,
                 active: Union[mi.Bool, bool] = True
    ) -> Tuple[mi.Vecto3f, mi.Float]:
        return self.reparam(self.scene, self.params, ray, active=active)

def eval_scene(
    params: mi.SceneParameters,
    p: mi.Vector3f,
    eps: mi.Float,
    active: mi.Bool,
) -> Tuple[mi.Float, mi.Bool, mi.Vector3f, mi.Color3f]:
    
    shape_idx = 0
    shape_dist = 1e99
    normal = mi.Vector3f(0)
    hessian = mi.Matrix3f(0)
    color = mi.Color3f(0)

    while True:
        reflectance = params.get(f'sphere{shape_idx}.bsdf.reflectance.value')
        if reflectance is None:
            break

        to_world = params.get(f'sphere{shape_idx}.to_world')

        sphere = SphereSDF(
            reflectance,
            to_world,
            mi.Float(1)
        )

        t, n, dfdxx = sphere.eval(p, active)

        select = t < shape_dist
        shape_dist = dr.select(select, t, shape_dist)
        normal = dr.select(select, n, normal)
        hessian = dr.select(select, dfdxx, hessian)
        color = dr.select(select, sphere.color, color)
        hit = dr.abs(shape_dist) < eps

        shape_idx += 1

    return shape_dist, hit, mi.Vector3f(normal), hessian, mi.Color3f(dr.select(hit, color, mi.Color3f(0)))

class SphereSDF:
    DRJIT_STRUCT = { 'color' : mi.Color3f, 'transform': mi.Transform4f, 'scale' : mi.Float }

    def __init__(self, color: mi.Color3f, transform: mi.Transform4f, scale: mi.Float):
        self.color = color
        self.transform = transform
        self.scale = scale

    def eval(self,
        position: mi.Point3f,
        active: mi.Bool):
        T_inv = self.transform.inverse()
        transformed_p = T_inv.transform_affine(position)
        p = transformed_p / self.scale
        norm_p = dr.norm(p)

        norm_grad = dr.select(dr.neq(norm_p, 0), p / norm_p, 0)
        normal = dr.transpose(mi.Matrix3f(T_inv.matrix))@norm_grad

        dfdx = dr.detach(normal)
        dfdx_outer = mi.Matrix3f(dfdx * dfdx[0], dfdx * dfdx[1], dfdx * dfdx[2])
        dfdxx = dr.select(dr.neq(norm_p, 0), (1 - dfdx_outer) / (norm_p * self.scale), 0)

        return self.scale * (norm_p - 1.0), normal, dfdxx

class SDFIntegrator(ADIntegrator):
    def __init__(self, props=...):
        super().__init__(props)

        self.ray_march_max_it =  props.get('ray_march_max_it', 32)
        self.ray_march_eps = props.get('ray_march_eps', 1e-3)

        self.reparam_𝜆d = props.get('reparam_𝜆d', 1e-1)
        self.reparam_exp = props.get('reparam_exp', 4)

    def reparam(self,
            scene: mi.Scene,
            params: mi.SceneParameters,
            ray: mi.Ray3f,
            active: mi.Bool):
        return reparameterize_ray(scene, params, ray,
                                    ray_march_max_it=self.ray_march_max_it,
                                    ray_march_eps=self.ray_march_eps,
                                    𝜆d=self.reparam_𝜆d,
                                    exponent=self.reparam_exp,
                                    active=active)

    def ray_march_scene(self,
        primal: mi.Bool,
        scene: mi.Scene,
        ray: mi.Ray3f,
        active: mi.Bool):
        
        p = mi.Point3f(ray.o)
        dp = dr.grad(p)
        n = mi.Vector3f(0)
        c = mi.Color3f(0)

        it = mi.UInt32(0)
        active = mi.Bool(active)

        loop = mi.Loop(name="ray marching",
                       state=lambda: (it, p, dp, n, c, active))
        
        params = mi.traverse(scene)

        while loop(active & (it < self.ray_march_max_it)):
            tmp_p = mi.Point3f(p)

            if not primal:
                dr.enable_grad(tmp_p)
                dr.set_grad(tmp_p, dp)

            t, hit, _, _, _ = eval_scene(params, tmp_p, self.ray_march_eps, active)

            if not primal:
                dp[active] += dr.forward_to(t * ray.d)

            with dr.suspend_grad(when=not primal):
                p[active] = mi.Point3f(p + t * ray.d)

            active &= ~hit
            it += 1

        if not primal:
            dr.enable_grad(p)
            dr.set_grad(p, dp)

        _, _, n, _, c = eval_scene(params, p, self.ray_march_eps, True)

        valid = ~active

        return dr.normalize(n), c, valid

    def sample(self,
        mode: dr.ADMode,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        active: mi.Bool,
        **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        primal = mode == dr.ADMode.Primal

        L = mi.Spectrum(0)

        normal, color, valid = self.ray_march_scene(primal, scene, ray, active)

        light_dir = dr.normalize(mi.Vector3f(0.5, 1, 1))
        L[valid] += color * dr.maximum(dr.dot(normal, light_dir), 0) + mi.Color3f(0.01, 0.01, 0.01)

        return L, active, None
    
    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        reparam: Callable[[mi.Ray3f, mi.Bool],
                          Tuple[mi.Ray3f, mi.Float]] = None
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray

        When a reparameterization function is provided via the 'reparam'
        argument, it will be applied to the returned image-space position (i.e.
        the sample positions will be moving). The other two return values
        remain detached.
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        ray, weight = sensor.sample_ray_differential(
            time=time,
            sample1=wavelength_sample,
            sample2=pos_adjusted,
            sample3=aperture_sample
        )

        reparam_det = 1.0

        if reparam is not None:
            if rfilter.is_box_filter():
                raise Exception(
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. This is, however, incompatible with the box "
                    "reconstruction filter that is currently used. Please "
                    "specify a a smooth reconstruction filter in your scene "
                    "description (e.g. 'gaussian', which is actually the "
                    "default)")

            # This is less serious, so let's just warn once
            if not film.sample_border() and self.sample_border_warning:
                self.sample_border_warning = True

                mi.Log(mi.LogLevel.Warn,
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. To correctly account for shapes entering "
                    "or leaving the viewport, it is recommended that you set "
                    "the film's 'sample_border' parameter to True.")

            with dr.resume_grad():
                # Reparameterize the camera ray
                reparam_d, reparam_det = reparam(ray=ray)

                # Create a fake interaction along the sampled ray and use it to the
                # position with derivative tracking
                it = dr.zeros(mi.Interaction3f)
                it.p = ray.o + reparam_d
                ds, _ = sensor.sample_direction(it, aperture_sample)

                # Return a reparameterized image position
                pos_f = ds.uv + film.crop_offset()

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos, reparam_det
    
    def render(self: SDFIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, state = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = sensor.film().create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))
            if mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special):
                aovs = sensor.film().prepare_sample(L * weight, ray.wavelengths,
                                                    block.channel_count(), alpha=alpha)
                block.put(pos, aovs)
                del aovs
            else:
                block.put(pos, ray.wavelengths, L * weight, alpha)

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid, alpha
            gc.collect()

            # Perform the weight division and return an image tensor
            sensor.film().put_block(block)
            self.primal_image = sensor.film().develop()

            return self.primal_image
        
    def render_forward(self: SDFIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # When the underlying integrator supports reparameterizations,
            # perform necessary initialization steps and wrap the result using
            # the _ReparamWrapper abstraction defined above
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            with dr.resume_grad():
                L, valid, _ = self.sample(
                    mode=dr.ADMode.Forward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    reparam=reparam,
                    active=mi.Bool(True)
                )

                block = film.create_block()
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                # Deposit samples with gradient tracking for 'pos'.
                # After reparameterizing the camera ray, we need to evaluate
                #   Σ (fi Li det)
                #  ---------------
                #   Σ (fi det)
                if (dr.all(mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special))):
                    aovs = sensor.film().prepare_sample(L * weight * det, ray.wavelengths,
                                                        block.channel_count(),
                                                        weight=det,
                                                        alpha=dr.select(valid, mi.Float(1), mi.Float(0)))
                    block.put(pos, aovs)
                    del aovs
                else:
                    block.put(
                        pos=pos,
                        wavelengths=ray.wavelengths,
                        value=L * weight * det,
                        weight=det,
                        alpha=dr.select(valid, mi.Float(1), mi.Float(0))
                    )

                # Perform the weight division and return an image tensor
                film.put_block(block)
                result_img = film.develop()

                dr.forward_to(result_img)

        return dr.grad(result_img)

    def render_backward(self: SDFIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # When the underlying integrator supports reparameterizations,
            # perform necessary initialization steps and wrap the result using
            # the _ReparamWrapper abstraction defined above
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            with dr.resume_grad():
                L, valid, _ = self.sample(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    active=mi.Bool(True)
                )

                # Prepare an ImageBlock as specified by the film
                block = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                if mi.has_flag(sensor.film().flags(), mi.FilmFlags.Special):
                    aovs = sensor.film().prepare_sample(L * weight * det, ray.wavelengths,
                                                        block.channel_count(),
                                                        weight=det,
                                                        alpha=dr.select(valid, mi.Float(1), mi.Float(0)))
                    block.put(pos, aovs)
                    del aovs
                else:
                    block.put(
                        pos=pos,
                        wavelengths=ray.wavelengths,
                        value=L * weight * det,
                        weight=det,
                        alpha=dr.select(valid, mi.Float(1), mi.Float(0))
                    )

                sensor.film().put_block(block)

                del valid
                gc.collect()

                # This step launches a kernel
                dr.schedule(block.tensor())
                image = sensor.film().develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)

            # We don't need any of the outputs here
            del ray, weight, pos, block, sampler
            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()


mi.register_integrator("sdf_integrator", lambda props: SDFIntegrator(props))