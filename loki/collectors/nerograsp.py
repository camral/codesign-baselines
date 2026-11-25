"""Native NeroGrasp mesh sampling with immutable topology and mountings."""

from .base import BaseCollector


class NeroGraspCollector(BaseCollector):
    default_sample_perturbation_m = 0.0003
    presets = ("NeroGrasp-All", "NeroGrasp-Small", "NeroGrasp-Tiny", "NeroGrasp-Thin",
               "NeroGrasp-SetScrew", "NeroGrasp-TNut", "NeroGrasp-LockNut", "NeroGrasp-Screw",
               "NeroGrasp-Scissors", "NeroGrasp-Tweezers", "NeroGrasp-Spatula", "NeroGrasp-Stick",
               "NeroGrasp-ScrewLong")

    def make_env(self):
        from codesign_gym import REGISTRY, make_vec
        from envs.nerograsp.envs.leap_grasp import NeroGraspEnv
        if self.device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("NeroGrasp selects cuda:0; use CUDA_VISIBLE_DEVICES to select a GPU")
        kwargs = {"seed": self.seed, "use_cpu": self.device == "cpu",
                  "sample_perturbation_m": self.default_sample_perturbation_m,
                  **self.env_kwargs}
        # Geometry validity and the fingertip design space are identical across
        # NeroGrasp tasks. Building NeroGrasp-All would repeat the same geometry
        # checks once per object, so the shared corpus uses one representative
        # object world for validation and reset audits.
        build_env_id = ("NeroGrasp-SetScrew"
                        if self.env_id == "NeroGrasp-All" else self.env_id)
        selector = kwargs.get(
            "object_name", REGISTRY[build_env_id][1]["object_name"])
        groups = NeroGraspEnv.TYPES_TO_OBJS
        worlds = (sum(map(len, groups.values())) if selector == "all"
                  else len(groups[selector]) if selector in groups else 1)
        return make_vec(build_env_id, worlds, is_eval_env=True, **kwargs)[0]

    def configure(self):
        if getattr(self.env, "num_embodiments", 1) != 1:
            raise ValueError("NeroGrasp collector requires exactly one logical embodiment slot")
        self.component_proposals = {key: 0 for key in self.space.base_meshes}

    def sample_embodiment(self):
        # Native validity is a conjunction of independent per-fingertip checks.
        # Sampling each fingertip from its own conditional-valid distribution
        # and then pairing them is exactly the original joint proposal
        # conditioned on validity, while avoiding the multiplicative rejection
        # rate from discarding a valid tip when its partner is invalid.
        design = self.space.base_meshes
        for key in design:
            for _ in range(10_000):
                proposal = self.space.sample(rng=self.rng)
                probe = self.space.base_meshes
                probe[key] = proposal[key]
                self.component_proposals[key] += 1
                valid, _ = super().validate_embodiment(probe)
                if valid:
                    design[key] = proposal[key]
                    break
            else:
                raise RuntimeError(
                    f"Could not sample a valid NeroGrasp {key} component")
        return design

    def validation_input(self, design):
        return [design]

    def sampler_metadata(self):
        return {"distribution": "independent native fingertip proposals conditioned on per-fingertip validity, then paired",
                "sample_perturbation_m": self.space.sample_perturbation_m,
                "topology": "fixed native template", "mountings": "unchanged",
                "validation": "full native validator, including enabled geometry checks; convex decomposition is deferred to reset",
                "representative_validation_world": ("NeroGrasp-SetScrew"
                                                    if self.env_id == "NeroGrasp-All"
                                                    else self.env_id)}
