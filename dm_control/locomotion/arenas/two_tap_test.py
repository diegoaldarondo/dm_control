from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control import composer
from dm_control.composer.variation import deterministic
from dm_control.locomotion.arenas import two_tap
from six.moves import zip
from dm_control import viewer
from dm_control.entities.props import primitive


class TwoTapTest(absltest.TestCase):
    def test_can_compile_mjcf(self):
        arena = two_tap.TwoTap()
        mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    def test_can_view(self):
        arena = two_tap.TwoTap()
        task = composer.NullTask(arena)
        env = composer.Environment(task)
        env.reset()
        # viewer.launch(env)

    def test_can_move_lever(self):
        arena = two_tap.TwoTap()
        prop = primitive.Primitive(
            geom_type="box",
            mass=1,
            contype=1,
            conaffinity=1,
            size=[0.002, 0.002, 0.002],
            pos=[0, -0.0075, 0.1],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
        )
        arena.add_free_entity(prop)
        task = composer.NullTask(arena)
        task.initialize_episode = lambda physics, random_state: prop.set_pose(
            physics, [0, -0.005, 0.05]
        )
        env = composer.Environment(task)
        env.reset()
        # viewer.launch(env)

    def test_can_two_tap(self):
        arena = two_tap.TwoTap()

        # First tap prop
        prop1 = primitive.Primitive(
            geom_type="box",
            mass=1,
            size=[0.002, 0.002, 0.002],
            pos=[0, -0.0125, 0.06],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
        )
        arena.add_free_entity(prop1)

        # Second tap prop
        prop2 = primitive.Primitive(
            geom_type="box",
            mass=1,
            size=[0.002, 0.002, 0.002],
            pos=[0, -0.007, 0.06],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
        )
        arena.add_free_entity(prop2)

        task = composer.NullTask(arena)
        def two_tap_after_step(physics, random_state):
          arena.lever.after_substep(physics, random_state)
          arena.spout._water._update_activation(physics)
        task.after_step = two_tap_after_step

        # Set the second prop vel so that it hits the lever later
        task.initialize_episode = lambda physics, random_state: prop2.set_velocity(
            physics, [0, 0, 1.3]
        )
        env = composer.Environment(task)
        env.reset()
        viewer.launch(env)

    def test_can_activate_lever(self):
        arena = two_tap.TwoTap()

        # First tap prop
        prop1 = primitive.Primitive(
            geom_type="box",
            mass=1,
            size=[0.002, 0.002, 0.002],
            pos=[0, -0.0125, 0.06],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
        )
        arena.add_free_entity(prop1)
        task = composer.NullTask(arena)
        def two_tap_after_step(physics, random_state):
          arena.lever.after_substep(physics, random_state)
          arena.spout._water._update_activation(physics)
        task.after_step = two_tap_after_step
        env = composer.Environment(task)
        env.reset()
        # viewer.launch(env)


if __name__ == "__main__":
    absltest.main()
