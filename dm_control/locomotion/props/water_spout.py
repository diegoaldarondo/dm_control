from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.props import target_sphere
import numpy as np


class Droplet(target_sphere.TargetSphere):
    def _build(
        self,
        radius=0.6,
        height_above_ground=0.0,
        rgba=(0.0, 0.0, 0.7, 0),
        specific_collision_geom_ids=None,
        name="droplet",
    ):
        self._mjcf_root = mjcf.RootElement(model=name)
        self._geom = self._mjcf_root.worldbody.add(
            "geom",
            type="sphere",
            name="geom",
            gap=2 * radius,
            pos=[0, 0, height_above_ground],
            size=[radius],
            rgba=rgba,
        )
        self._geom_id = -1
        self._activated = False
        self._specific_collision_geom_ids = specific_collision_geom_ids

    def _update_activation(self, physics):
        if self._activated:
            physics.bind(self._geom).rgba[-1] = 1
            for contact in physics.data.contact:
                if self._specific_collision_geom_ids:
                    has_specific_collision = (
                        contact.geom1 in self._specific_collision_geom_ids
                        or contact.geom2 in self._specific_collision_geom_ids
                    )
                else:
                    has_specific_collision = True
                if has_specific_collision and self._geom_id in (
                    contact.geom1,
                    contact.geom2,
                ):
                    self._activated = False
                    physics.bind(self._geom).rgba[-1] = 0

    def reset(self, physics):
        self._activated = False
        physics.bind(self._geom).rgba[-1] = 0

    def initialize_episode_mjcf(self, unused_random_state):
        self._activated = False


class WaterSpout(composer.Entity):
    def _build(self, name, pos=[0, 0, 0], size=[0.001, 0.014], water_radius=0.002):
        self._mjcf_root = mjcf.RootElement(model=name)
        self._spout = self.mjcf_model.worldbody.add(
            "body", name=name, pos=pos, xyaxes=[1, 0, 0, 0, 0, 1]
        )
        self._geom = self._spout.add(
            "geom",
            name=name,
            type="capsule",
            size=size,
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
            solimp=[0.99, 0.9999, 0.001],
        )

        # Build the water droplet
        water_site = self._mjcf_root.worldbody.add(
            "site", size=[1e-6] * 3, pos=[0.0, -0.014, 0.0]
        )

        self._water = Droplet(radius=water_radius,)
        water_site.attach(self._water.mjcf_model)

    @property
    def geom(self):
        return self._geom

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def has_droplet(self):
        return self._water._activated

    def release_droplet(self):
        self._water._activated = True
