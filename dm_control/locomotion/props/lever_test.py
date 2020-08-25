# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for props.lever."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from dm_control import composer

from dm_control.entities.props import primitive
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.props import lever
from dm_control import viewer
import numpy as np


class LeverTest(absltest.TestCase):
    def setup_arena_and_one_tap(self):
        arena = floors.Floor(size=(0.1, 0.1))
        prop = primitive.Primitive(
            geom_type="box",
            mass=1,
            contype=1,
            conaffinity=1,
            size=[0.002, 0.002, 0.002],
            pos=[0, -0.01, 0.15],
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
        )
        arena.add_free_entity(prop)
        return arena, prop

    def setup_arena_and_two_tap(self):
        # First tap prop
        arena = floors.Floor(size=(0.1, 0.1))
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
        return arena, (prop1, prop2)

    def setup_lever_env(self, setup_function, lever_class, lever_kwargs={}):
        arena, props = setup_function()

        lever_site = arena._mjcf_root.worldbody.add(
            "site", size=[1e-6] * 3, pos=[0, 0., 0.055]
        )

        lever_prop = lever_class(**lever_kwargs)
        lever_site.attach(lever_prop.mjcf_model)

        task = composer.NullTask(arena)
        env = composer.Environment(task)
        task.after_step = lambda physics, random_state: lever_prop.after_substep(
            physics, random_state
        )
        env.reset()
        return env, lever_prop, props

    def activate_and_reset_lever(self, env, lever_prop):
        # Test base state
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > lever_prop._trigger_position
        ):
            self.assertFalse(lever_prop.pressed)
            env.step([])

        # Test activated when triggered and color changed
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > lever_prop._trigger_position
        ):
            self.assertFalse(lever_prop.pressed)
            env.step([])

        # Target should be reset when the environment is reset.
        env.reset()
        self.assertFalse(lever_prop.pressed)

    def test_lever_activates(self):
        env, lever_prop, _ = self.setup_lever_env(
            setup_function=self.setup_arena_and_one_tap, lever_class=lever.Lever
        )
        self.activate_and_reset_lever(env, lever_prop)

    def test_two_tap_lever_activates(self):
        env, lever_prop, _ = self.setup_lever_env(
            setup_function=self.setup_arena_and_one_tap, lever_class=lever.TwoTapLever
        )
        self.activate_and_reset_lever(env, lever_prop)

    def test_two_tap_lever_pass(self):
        env, lever_prop, props = self.setup_lever_env(
            setup_function=self.setup_arena_and_two_tap,
            lever_class=lever.TwoTapLever,
            lever_kwargs={"tap_interval": [0.2, 0.4], "trigger_position": np.pi / 8},
        )

        # Set the second prop vel so that it hits the lever later
        def init_func(physics, random_state):
            props[1].set_velocity(physics, [0, 0, 1.3])
            lever_prop.reset(physics)

        env.task.initialize_episode = init_func
        env.reset()

        # Initial base state
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.BASE)
            env.step([])

        # First tap
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.TAP1)
            env.step([])

        # Lever above trigger, first tap
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.TAP1)
            env.step([])

        # Successful second tap
        if (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.PASS)
            env.step([])

        # Inter trial interval
        while (
            env.physics.time() - lever_prop._trigger_time
        ) < lever_prop.inter_trial_interval:
            self.assertTrue(lever_prop._state == lever.LeverState.PASS)
            env.step([])

        # State resets to base state after inter trial interval
        if (
            env.physics.time() - lever_prop._trigger_time
        ) > lever_prop.inter_trial_interval:
            self.assertTrue(lever_prop._state == lever.LeverState.BASE)
            env.step([])

    def test_two_tap_lever_fail_early(self):
        env, lever_prop, props = self.setup_lever_env(
            setup_function=self.setup_arena_and_two_tap,
            lever_class=lever.TwoTapLever,
            lever_kwargs={"tap_interval": [0.7, 0.8], "trigger_position": np.pi / 8},
        )

        # Set the second prop vel so that it hits the lever later
        def init_func(physics, random_state):
            props[1].set_velocity(physics, [0, 0, 1.3])
            lever_prop.reset(physics)

        env.task.initialize_episode = init_func
        env.reset()
        viewer.launch(env)
        # Initial base state
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.BASE)
            env.step([])

        # First tap
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.TAP1)
            env.step([])

        # Lever above trigger, first tap
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.TAP1)
            env.step([])

        # Early trigger fails
        if (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            > lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.FAIL)
            env.step([])

        # Penalty time
        while (
            env.physics.time() - lever_prop._trigger_time
        ) < lever_prop.penalty_interval:
            self.assertTrue(lever_prop._state == lever.LeverState.FAIL)
            env.step([])

        # State resets to base state after penalty interval
        if (
            env.physics.time() - lever_prop._trigger_time
        ) > lever_prop.penalty_interval:
            self.assertTrue(lever_prop._state == lever.LeverState.BASE)
            env.step([])

    def test_two_tap_lever_fail_late(self):
        env, lever_prop, props = self.setup_lever_env(
            setup_function=self.setup_arena_and_two_tap,
            lever_class=lever.TwoTapLever,
            lever_kwargs={"tap_interval": [0.1, 0.11], "trigger_position": np.pi / 8},
        )

        # Set the second prop vel so that it hits the lever later
        def init_func(physics, random_state):
            props[1].set_velocity(physics, [0, 0, 1.3])
            lever_prop.reset(physics)

        env.task.initialize_episode = init_func
        env.reset()
        # Initial base state
        while (
            np.copy(env.physics.named.data.qpos["lever/lever_hinge"])
            < lever_prop._trigger_position
        ):
            self.assertTrue(lever_prop._state == lever.LeverState.BASE)
            env.step([])

        # First tap
        while (
            env.physics.time() - lever_prop._previous_trigger_time
        ) < lever_prop.tap_interval[1]:
            self.assertTrue(lever_prop._state == lever.LeverState.TAP1)
            env.step([])

        # Time expires
        if (
            env.physics.time() - lever_prop._previous_trigger_time
        ) > lever_prop.tap_interval[1]:
            self.assertTrue(lever_prop._state == lever.LeverState.FAIL)
            env.step([])

        # Penalty time
        while (
            env.physics.time() - lever_prop._trigger_time
        ) < lever_prop.penalty_interval:
            self.assertTrue(lever_prop._state == lever.LeverState.FAIL)
            env.step([])

        # State resets to base state after penalty interval
        if (
            env.physics.time() - lever_prop._trigger_time
        ) > lever_prop.penalty_interval:
            self.assertTrue(lever_prop._state == lever.LeverState.BASE)
            env.step([])


if __name__ == "__main__":
    absltest.main()
