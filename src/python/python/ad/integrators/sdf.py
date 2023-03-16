from __future__ import annotations
from typing import Tuple, Dict # Delayed parsing of type annotations
import math

import drjit as dr
import mitsuba as mi
import gc

from .common import ADIntegrator

def abs_grad(x):
    return dr.select(x >= 0, 1, -1)

def normalize_grad(v):
    norm = dr.norm(v)
    inv_norm = dr.select(dr.neq(norm, 0), 1 / norm, 0)
    s = v * inv_norm
    sos = mi.Matrix3f(s * s[0], s * s[1], s * s[2])
    return inv_norm * (1 - sos)

def _sample_warp_field(sensor: mi.Sensor,
                       params: mi.SceneParameters,
                       u: mi.Vector2f,
                       pos: mi.Vector3f,
                       dir: mi.Vector3f,
                       t: mi.Float,
                       ray_march_eps: mi.Float,
                       𝜆d: mi.Float,
                       exponent: mi.Float,
                       active: mi.Bool):
    f, hit, dfdx, dfdxx, _ = eval_scene(params, pos, ray_march_eps, active)

    sensor_params = mi.traverse(sensor)
    camera_to_sample = mi.perspective_projection(
        sensor.film().size(),
        sensor.film().crop_size(),
        sensor.film().crop_offset(),
        sensor_params['x_fov'],
        sensor_params['near_clip'],
        sensor_params['far_clip'])
    sample_to_camera = camera_to_sample.inverse()
    camera_to_world = sensor_params['to_world']

    with dr.suspend_grad():
        sil_dot = dr.dot(dfdx, dir)
        S = dr.abs(f) + 𝜆d * dr.abs(sil_dot)
        w = dr.select(S > 1e-4, dr.power(dr.rcp(S), exponent), 0.0)

        near_p = sample_to_camera @ mi.Point3f(u.x, u.y, 0)
        ddir_du = mi.Matrix3f(camera_to_world.matrix) \
            @ normalize_grad(near_p) \
            @ dr.transpose(mi.Matrix3f(sample_to_camera.matrix)) \
            / sample_to_camera.matrix[3,3]
        
        dx_du = t * ddir_du
        dx_du_t = dr.transpose(dx_du)
        du_dx = dr.select(dr.neq(t, 0), mi.Matrix3f(dr.inverse(mi.Matrix2f(dx_du_t @ dx_du))) @ dx_du_t, 0)

        da_df = abs_grad(f)
        da_dsildot = abs_grad(sil_dot)

        ds_du = da_df * dfdx @ dx_du + 𝜆d * da_dsildot * (dfdx @ ddir_du + dir @ dfdxx @ dx_du)
        ds_du = mi.Vector2f(ds_du.x, ds_du.y)
        dw_du = -exponent * dr.select(S > 1e-4, dr.power(dr.rcp(S), exponent+1), 0.0) * ds_du

    V_direct = -f * dr.detach(dfdx) @ du_dx
    V_direct = mi.Vector2f(V_direct.x, V_direct.y)

    return w, dw_du, w * V_direct, dr.dot(dw_du, V_direct), dr.detach(f), hit

class _ReparameterizeOp(dr.CustomOp):
    def eval(self, scene, sensor, params, pos, ray_march_max_it, ray_march_eps,
             𝜆d, exponent, active):
        # Stash all of this information for the forward/backward passes
        self.scene = scene
        self.sensor = sensor
        self.params = params
        self.pos = dr.detach(pos)
        self.ray_march_max_it = ray_march_max_it
        self.ray_march_eps = ray_march_eps
        self.𝜆d = 𝜆d
        self.exponent = exponent
        self.active = active

        # The reparameterization is simply the identity in primal mode
        return self.pos, dr.full(mi.Float, 1, dr.width(pos))


    def forward(self):
        """
        Propagate the gradients in the forward direction to 'ray.d' and the
        jacobian determinant 'det'. From a warp field point of view, the
        derivative of 'ray.d' is the warp field direction at 'ray', and
        the derivative of 'det' is the divergence of the warp field at 'ray'.
        """

        # Initialize some accumulators
        Z = mi.Float(0.0)
        dZ = mi.Vector2f(0.0)
        grad_V = mi.Vector2f(0.0)
        grad_div_lhs = mi.Float(0.0)
        it = mi.UInt32(0)
        active = mi.Bool(self.active)
        
        ray, _, _ = sample_rays_from_screen_pos(self.pos, self.sensor)
        f, hit, _, _, _ = eval_scene(self.params, ray.o, self.ray_march_eps, active)
        t_prev = dr.detach(f)
        t_delta = dr.detach(f)
        active &= ~hit

        pos_grad = self.grad_in('pos')

        loop = mi.Loop(name="reparameterize_screen_pos(): forward propagation",
                       state=lambda: (it, active, t_delta, t_prev, Z, dZ, grad_V, grad_div_lhs))

        # Unroll the entire loop in wavefront mode
        # loop.set_uniform(True) # TODO can we turn this back on? (see self.active in loop condition)
        loop.set_max_iterations(self.ray_march_max_it)
        loop.set_eval_stride(self.ray_march_max_it)

        while loop(active & (it < self.ray_march_max_it)):
            u = mi.Vector2f(self.pos)
            dr.enable_grad(u)
            dr.set_grad(u, pos_grad)
            ray, _, _ = sample_rays_from_screen_pos(u, self.sensor)

            p = ray.o + t_prev * ray.d

            Z_i, dZ_i, V_i, div_lhs_i, t, hit = _sample_warp_field(
                self.sensor,
                self.params,
                u,
                p,
                ray.d,
                t_prev,
                self.ray_march_eps,
                self.𝜆d,
                self.exponent,
                active)

            # Do not clear input vertex gradient
            dr.forward_to(V_i, div_lhs_i,
                          flags=dr.ADFlag.ClearEdges | dr.ADFlag.ClearInterior)

            Z += Z_i * t_delta
            dZ += dZ_i * t_delta
            grad_V += dr.grad(V_i) * t_delta
            grad_div_lhs += dr.grad(div_lhs_i) * t_delta

            t_delta = t
            t_prev += t
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
        pass

    def name(self):
        return "reparameterize_screen_pos()"


def reparameterize_screen_pos(scene: mi.Scene,
                              sensor: mi.Sensor,
                              params: mi.SceneParameters,
                              pos: mi.Vector2f,
                              ray_march_max_it: int=32,
                              ray_march_eps=1e-3,
                              𝜆d: float=1e-1,
                              exponent: float=4.0,
                              active: mi.Bool = True
) -> Tuple[mi.Vector2f, mi.Float]:
    return dr.custom(_ReparameterizeOp, scene, sensor, params, pos,
                     ray_march_max_it, ray_march_eps, 𝜆d, exponent, active)


class _ReparamWrapper:
    # ReparamWrapper instances can be provided as dr.Loop state
    # variables. For this to work we must declare relevant fields
    DRJIT_STRUCT = { }

    def __init__(self,
                 scene : mi.Scene,
                 sensor: mi.Sensor,
                 params: Any,
                 reparam: Callable[
                     [mi.Scene, mi.Sensor, mi.SceneParameters, mi.Vector2f,
                      float, float, mi.Bool],
                     Tuple[mi.Vector2f, mi.Float]]):

        self.scene = scene
        self.sensor = sensor
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
                 pos: mi.Vector2f,
                 active: Union[mi.Bool, bool] = True
    ) -> Tuple[mi.Vector2f, mi.Float]:
        return self.reparam(self.scene, self.sensor, self.params, pos, active=active)


def sample_rays_from_screen_pos(
    pos_f: mi.Vector2f,
    sensor: mi.Sensor,
) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f]:
    film = sensor.film()
    rfilter = film.rfilter()

    # Re-scale the position to [0, 1]^2
    scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale
    pos_adjusted = dr.fma(pos_f, scale, offset)

    with dr.resume_grad():
        ray, weight = sensor.sample_ray_differential(
            time=0,
            sample1=0,
            sample2=pos_adjusted,
            sample3=mi.Point2f(0, 0)
        )

    # With box filter, ignore random offset to prevent numerical instabilities
    splatting_pos = mi.Vector2f(mi.Vector2d(pos_f)) if rfilter.is_box_filter() else pos_f

    return ray, weight, splatting_pos

def eval_scene(
    params: mi.SceneParameters,
    p: mi.Vector3f,
    eps: mi.Float,
    active: mi.Bool,
) -> Tuple[mi.Float, mi.Bool, mi.Vector3f, mi.Color3f]:
    color = params.get('sphere.bsdf.reflectance.value')
    to_world = params.get('sphere.to_world')

    sphere = SphereSDF(
        color,
        to_world,
        mi.Float(1)
    )

    t, n, dfdxx = sphere.eval(p, active)
    hit = dr.abs(t) < eps

    return t, hit, mi.Vector3f(n), dfdxx, mi.Color3f(dr.select(hit, sphere.color, mi.Color3f(0)))

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

        self.reparam_𝜆d = props.get('reparam_kappa', 1e-1)
        self.reparam_exp = props.get('reparam_exp', 4)

    def reparam(self,
            scene: mi.Scene,
            sensor: mi.Sensor,
            params: mi.SceneParameters,
            pos: mi.Vector2f,
            active: mi.Bool):
        return reparameterize_screen_pos(scene, sensor, params, pos,
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

        return n, c, valid

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
    
    def sample_screen_pos(self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        reparam: Callable[[mi.Vector2f, mi.Bool],
                          Tuple[mi.Vector2f, mi.Float]] = None
    ) -> Tuple[mi.Vector2f, mi.Float]:
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

        reparam_det = 1.0

        if reparam is not None:
            if rfilter.is_box_filter():
                raise Exception(
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. This is, however, incompatible with the box "
                    "reconstruction filter that is currently used. Please "
                    "specify a smooth reconstruction filter in your scene "
                    "description (e.g. 'gaussian', which is actually the "
                    "default)")

            # This is less serious, so let's just warn once
            if not film.sample_border():
                mi.Log(mi.LogLevel.Warn,
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. To correctly account for shapes entering "
                    "or leaving the viewport, it is recommended that you set "
                    "the film's 'sample_border' parameter to True.")

            with dr.resume_grad():
                # Reparameterize the camera ray
                pos_f, reparam_det = reparam(pos=dr.detach(pos_f))

        return pos_f, reparam_det
    
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
            pos_f, _ = self.sample_screen_pos(scene, sensor, sampler)
            ray, weight, pos = sample_rays_from_screen_pos(pos_f, sensor)

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
                    sensor=sensor,
                    params=params,
                    reparam=self.reparam
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            pos_f, det = self.sample_screen_pos(scene, sensor, sampler, reparam)
            ray, weight, pos = sample_rays_from_screen_pos(pos_f, sensor)

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
                    sensor=sensor,
                    params=params,
                    reparam=self.reparam
                )
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            pos_f, det = self.sample_screen_pos(scene, sensor, sampler, reparam)
            ray, weight, pos = sample_rays_from_screen_pos(pos_f, sensor)

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