from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.props import lever
from dm_control.locomotion.props import water_spout
import numpy as np

_CORRIDOR_X_PADDING = 2.0
_WALL_THICKNESS = 0.16
_SIDE_WALL_HEIGHT = 4.0
_DEFAULT_ALPHA = 0.25


class Barrier(composer.Entity):
    def _build(self, name, xyaxes, size=[0.025, 0.025, 0.002]):
        self._mjcf_root = mjcf.RootElement(model=name)
        self._barrier = self.mjcf_model.worldbody.add("body", name=name, xyaxes=xyaxes)
        self._geom = self._barrier.add(
            "geom", name=name, type="box", size=size, rgba=[1, 1, 1, _DEFAULT_ALPHA]
        )

    @property
    def geom(self):
        return self._geom

    @property
    def mjcf_model(self):
        return self._mjcf_root


class TwoTap(composer.Arena):
    def _build(self, visible_side_planes=True, name="empty_corridor", lever_kwargs={}):
        """Builds the corridor.

    Args:
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      name: The name of this arena.
    """
        super(TwoTap, self)._build(name=name)
        self._mjcf_root.visual.map.znear = 0.0005
        self._mjcf_root.asset.add(
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1=[0.4, 0.6, 0.8],
            rgb2=[0, 0, 0],
            width=100,
            height=600,
        )
        self._mjcf_root.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

        alpha = _DEFAULT_ALPHA if visible_side_planes else 0.0
        self._ground_plane = self._mjcf_root.worldbody.add(
            "geom", type="plane", rgba=[0.5, 0.5, 0.5, 1], size=[0.5, 0.5, 0.5]
        )
        self._right_plane = self._mjcf_root.worldbody.add(
            "geom",
            contype=2,
            conaffinity=2,
            type="box",
            name="back_wall",
            xyaxes=[-1, 0, 0, 0, 0, 1],
            size=[0.5, 0.25, 0.001],
            rgba=[1, 1, 1, .05],
            pos=[0, 0, 0.25],
        )

        lever_site = self._mjcf_root.worldbody.add(
            "site", size=[1e-6] * 3, pos=[0, -0., 0.13]
        )
        self.lever = lever.TwoTapLever(name="lever", **lever_kwargs)
        lever_site.attach(self.lever.mjcf_model)

        spout_site = self._mjcf_root.worldbody.add(
            "site", size=[1e-6] * 3, pos=[0, -0.014, 0.055]
        )
        self.spout = water_spout.WaterSpout(name="spout")
        spout_site.attach(self.spout.mjcf_model)

        left_barrier_site = self._mjcf_root.worldbody.add(
            "site", size=[1e-6] * 3, pos=[-0.015, -0.025, 0.1275]
        )
        self.left_barrier = Barrier("left_barrier", [0.1, -1, 0, 0, 0, 1])
        left_barrier_site.attach(self.left_barrier.mjcf_model)

        right_barrier_site = self._mjcf_root.worldbody.add(
            "site", size=[1e-6] * 3, pos=[0.015, -0.025, 0.1275]
        )
        self.right_barrier = Barrier("right_barrier", [-0.1, -1, 0, 0, 0, 1])
        right_barrier_site.attach(self.right_barrier.mjcf_model)

    def regenerate(self, random_state):
        pass
