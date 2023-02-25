from __future__ import annotations
from typing import Tuple, Dict # Delayed parsing of type annotations
import math

import drjit as dr
import mitsuba as mi
import gc

from .common import ADIntegrator

class SphereSDF:
    DRJIT_STRUCT = { 'color' : mi.Color3f, 'transform': mi.Transform4f, 'scale' : mi.Float }

    def __init__(self, color: mi.Color3f, transform: mi.Transform4f, scale: mi.Float):
        self.color = color
        self.transform = transform
        self.scale = scale

    def eval(self,
        position: mi.Point3f,
        active: mi.Bool):
        transformed_p = self.transform.inverse().transform_affine(position)
        p = transformed_p / self.scale
        return self.scale * (dr.norm(p) - 1.0), dr.normalize(mi.Vector3f(transformed_p))

class SDFIntegrator(ADIntegrator):
    def __init__(self, props=...):
        super().__init__(props)

        self.eps = mi.Float(1e-3)
        self.max_it = mi.UInt32(64)

    def ray_march_scene(self,
        primal: mi.Bool,
        scene: mi.Scene,
        ray: mi.Ray3f,
        active: mi.Bool):
        
        p = mi.Point3f(ray.o)
        n = mi.Vector3f(0)
        c = mi.Color3f(0)

        it = mi.UInt32(0)
        active = mi.Bool(active)

        loop = mi.Loop(name="ray marching",
                       state=lambda: (it, p, n, c, active))
        
        params = mi.traverse(scene)
        color = params.get('sphere.bsdf.reflectance.value')
        to_world = params.get('sphere.to_world')

        sphere = SphereSDF(
            color,
            to_world,
            mi.Float(1)
        )

        with dr.suspend_grad(when=not primal):
            while loop(active & (it < self.max_it)):
                t, n = sphere.eval(p, active)
                hit = dr.abs(t) < self.eps

                p[active] = mi.Point3f(p + t * ray.d)

                active &= ~hit
                it[active] += 1

        if not primal:
            _, n = sphere.eval(p, active)

        valid = dr.neq(it, self.max_it)
        c = dr.select(valid, sphere.color, c)

        return n, c, valid

    def sample(self,
        mode: dr.ADMode,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        reparam: Optional[
            Callable[[mi.Ray3f, mi.Bool],
                    Tuple[mi.Ray3f, mi.Float]]],
        active: mi.Bool,
        **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        primal = mode == dr.ADMode.Primal

        L = mi.Spectrum(0)

        normal, color, valid = self.ray_march_scene(primal, scene, ray, active)

        light_dir = dr.normalize(mi.Vector3f(0.5, 1, 1))
        L[valid] += color * dr.maximum(dr.dot(normal, light_dir), 0) + mi.Color3f(0.1, 0.1, 0.1)

        return L, active, None
    
    def render(self: mi.SamplingIntegrator,
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
                reparam=None,
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

    def render_backward(self: mi.SamplingIntegrator,
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
            if False: #hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
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
                    reparam=reparam,
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