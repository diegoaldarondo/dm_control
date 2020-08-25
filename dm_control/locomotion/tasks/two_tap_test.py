"""Tests for locomotion.tasks.two_tap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import absltest

from dm_control import composer
import dm_control.locomotion.arenas.two_tap as two_tap_arena
from dm_control.locomotion.props import target_sphere, lever
from dm_control.locomotion.tasks import two_tap
from dm_control.locomotion.walkers import rodent
from dm_control.entities.props import primitive


import numpy as np
from dm_control import viewer

_CONTROL_TIMESTEP = 0.001
_PHYSICS_TIMESTEP = 0.001


class TwoTapTest(absltest.TestCase):
    def setup_props(self, arena):

        prop1 = primitive.Primitive(
            geom_type="box",
            mass=1,
            size=[0.001, 0.001, 0.001],
            pos=[0, -0.013, 0.136],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
            solimp=[0.99, 0.9999, 0.001],
        )
        arena.add_free_entity(prop1)

        # Second tap prop
        prop2 = primitive.Primitive(
            geom_type="box",
            mass=1,
            size=[0.001, 0.001, 0.001],
            pos=[0, -0.01, 0.136],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
            solimp=[0.99, 0.9999, 0.001],
        )
        arena.add_free_entity(prop2)
        return (prop1, prop2)

    def setup_task(self, lever_kwargs={}):
        walker = rodent.Rat()

        arena = two_tap_arena.TwoTap(lever_kwargs=lever_kwargs)

        task = two_tap.TwoTap(
            walker=walker,
            arena=arena,
            randomize_spawn_rotation=False,
            target_type_rewards=[25.0, 25.0, 25.0, 50],
            physics_timestep=_PHYSICS_TIMESTEP,
            control_timestep=_CONTROL_TIMESTEP,
        )
        return walker, arena, task

    def test_observables(self):
        walker, arena, task = self.setup_task()
        props = self.setup_props(arena)

        random_state = np.random.RandomState(12345)
        env = composer.Environment(task, random_state=random_state)
        timestep = env.reset()

        self.assertIn("walker/joints_pos", timestep.observation)

    def test_pass(self):
        walker, arena, task = self.setup_task(
            lever_kwargs={"tap_interval": [0.2, 0.4], "trigger_position": np.pi / 8}
        )
        props = self.setup_props(arena)

        random_state = np.random.RandomState(12345)
        env = composer.Environment(task, random_state=random_state)

        # Set the second prop vel so that it hits the lever later
        previous_init_func = env.task.initialize_episode
        def init_func(physics, random_state):
            previous_init_func(physics, random_state)
            props[1].set_velocity(physics, [0, 0, 1.3])
            arena.lever.reset(physics)
            arena.spout._water.reset(physics)

        env.task.initialize_episode = init_func
        action = np.zeros_like(env.physics.data.ctrl)

        # Initial base state
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < arena.lever._trigger_position
        ):
            self.assertTrue(arena.lever._state == lever.LeverState.BASE)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.BASE)
            env.step(action)

        # First tap
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > arena.lever._trigger_position
        ):
            self.assertTrue(arena.lever._state == lever.LeverState.TAP1)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.TAP1)
            env.step(action)

        # Lever above trigger, first tap
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < arena.lever._trigger_position
        ):
            self.assertTrue(arena.lever._state == lever.LeverState.TAP1)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.TAP1)
            env.step(action)

        # Successful second tap
        if (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > arena.lever._trigger_position
        ):
            self.assertTrue(arena.lever._state == lever.LeverState.PASS)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.PASS)
            self.assertTrue(arena.spout.has_droplet())
            env.step(action)

        # Inter trial interval
        while (
            env.physics.time() - arena.lever._trigger_time
        ) < arena.lever.inter_trial_interval:
            self.assertTrue(arena.lever._state == lever.LeverState.PASS)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.PASS)
            env.step(action)

        # State resets to base state after inter trial interval
        if (
            env.physics.time() - arena.lever._trigger_time
        ) > arena.lever.inter_trial_interval:
            self.assertTrue(arena.lever._state == lever.LeverState.BASE)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.BASE)
            env.step(action)

    def test_fail(self):
        walker, arena, task = self.setup_task(
            lever_kwargs={"tap_interval": [0.1, 0.11], "trigger_position": np.pi / 8}
        )
        props = self.setup_props(arena)

        random_state = np.random.RandomState(12345)
        env = composer.Environment(task, random_state=random_state)

        # Set the second prop vel so that it hits the lever later
        previous_init_func = env.task.initialize_episode
        def init_func(physics, random_state):
            previous_init_func(physics, random_state)
            props[1].set_velocity(physics, [0, 0, 1.3])
            arena.lever.reset(physics)
            arena.spout._water.reset(physics)

        env.task.initialize_episode = init_func
        env.reset()
        action = np.zeros_like(env.physics.data.ctrl)

        
        # Initial base state
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < arena.lever._trigger_position
        ):
            self.assertTrue(arena.lever._state == lever.LeverState.BASE)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.BASE)
            env.step(action)

        # First tap
        while (
            env.physics.time() - arena.lever._previous_trigger_time
        ) < arena.lever.tap_interval[1]:
            self.assertTrue(arena.lever.state == lever.LeverState.TAP1)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.TAP1)
            env.step(action)

        # Time expires
        if (
            env.physics.time() - arena.lever._previous_trigger_time
        ) > arena.lever.tap_interval[1]:
            self.assertTrue(arena.lever.state == lever.LeverState.FAIL)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.FAIL)
            env.step(action)

        # Penalty time
        while (
            env.physics.time() - arena.lever._trigger_time
        ) < arena.lever.penalty_interval:
            self.assertTrue(arena.lever.state == lever.LeverState.FAIL)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.FAIL)
            env.step(action)

        # State resets to base state after penalty interval
        if (
            env.physics.time() - arena.lever._trigger_time
        ) > arena.lever.penalty_interval:
            self.assertTrue(arena.lever.state == lever.LeverState.BASE)
            self.assertTrue(env.task._state_logic == two_tap.TwoTapState.BASE)
            env.step(action)


if __name__ == "__main__":
    absltest.main()
