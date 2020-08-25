# Copyright 2020 The dm_control Authors.
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
"""A (visuomotor) task consisting of reaching to targets for reward."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from dm_control import composer
from dm_control.composer.observation import observable as dm_observable
from dm_control.locomotion.props.lever import LeverState
import enum
import numpy as np
from six.moves import range
from six.moves import zip

DEFAULT_ALIVE_THRESHOLD = -1.0
DEFAULT_PHYSICS_TIMESTEP = 0.005
DEFAULT_CONTROL_TIMESTEP = 0.03


class TwoTapState(enum.IntEnum):
    BASE = 0
    TAP1 = 1
    PASS = 2  # Passed, water available
    PASS_WATER_CONSUMED = 3  # Passed, water consumed
    FAIL = 4


class TwoTap(composer.Task):
    """Task with target to tap with short delay (for Rat)."""

    def __init__(
        self,
        walker,
        arena,
        target_type_rewards,
        randomize_spawn_position=False,
        randomize_spawn_rotation=False,
        rotation_bias_factor=0,
        aliveness_reward=0.0,
        touch_interval=0.7,
        interval_tolerance=0.1,  # consider making a curriculum
        failure_timeout=2,
        reset_delay=1.2,
        physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
        control_timestep=DEFAULT_CONTROL_TIMESTEP,
    ):
        self._walker = walker
        self._arena = arena
        self._walker.create_root_joints(self._arena.attach(self._walker))
        if "CMUHumanoid" in str(type(self._walker)):
            self._lhand_body = walker.mjcf_model.find("body", "lhand")
            self._rhand_body = walker.mjcf_model.find("body", "rhand")
            self._head_body = walker.mjcf_model.find("body", "head")

        elif "Rat" in str(type(self._walker)):
            self._lhand_body = walker.mjcf_model.find("body", "hand_L")
            self._rhand_body = walker.mjcf_model.find("body", "hand_R")
            self._head_body = walker.mjcf_model.find("body", "jaw")
        else:
            raise ValueError("Expects Rat or CMUHumanoid.")

        self._lhand_geoms = self._lhand_body.find_all("geom")
        self._rhand_geoms = self._rhand_body.find_all("geom")
        self._head_geoms = self._head_body.find_all("geom")

        self._targets = []
        self._target_type_rewards = tuple(target_type_rewards)

        self._spawn_position = [0.0, -0.15]  # x, y
        self._rotation_bias_factor = rotation_bias_factor

        self._aliveness_reward = aliveness_reward
        self._discount = 1.0

        self._touch_interval = touch_interval
        self._interval_tolerance = interval_tolerance
        self._failure_timeout = failure_timeout
        self._reset_delay = reset_delay
        self._state_logic = TwoTapState.BASE
        self._randomize_spawn_rotation = randomize_spawn_rotation
        self._randomize_spawn_position = randomize_spawn_position

        self.set_timesteps(
            physics_timestep=physics_timestep, control_timestep=control_timestep
        )

        self._task_observables = collections.OrderedDict()

        def task_state(physics):
            del physics
            return np.array([self._state_logic])

        self._task_observables["task_logic"] = dm_observable.Generic(task_state)
        self._walker.observables.egocentric_camera.height = 64
        self._walker.observables.egocentric_camera.width = 64

        for observable in (
            self._walker.observables.proprioception
            + self._walker.observables.kinematic_sensors
            + self._walker.observables.dynamic_sensors
            + list(self._task_observables.values())
        ):
            observable.enabled = True
        self._walker.observables.egocentric_camera.enabled = True

    @property
    def name(self):
        return "two_touch"

    @property
    def task_observables(self):
        return self._task_observables

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        self._arena.regenerate(random_state)

    def _respawn_walker(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

        if self._randomize_spawn_position:
            self._spawn_position = self._arena.spawn_positions[
                random_state.randint(0, len(self._arena.spawn_positions))
            ]

        if self._randomize_spawn_rotation:
            rotation = 2 * np.pi * np.random.uniform()
            quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
        else:
            rotation = np.pi / 2
            quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]

        self._walker.shift_pose(
            physics,
            [self._spawn_position[0], self._spawn_position[1], 0.0],
            quat,
            rotate_velocity=True,
        )

    def initialize_episode(self, physics, random_state):
        super(TwoTap, self).initialize_episode(physics, random_state)
        self._respawn_walker(physics, random_state)
        self._state_logic = TwoTapState.BASE
        self._discount = 1.0
        self._lhand_geomids = set(physics.bind(self._lhand_geoms).element_id)
        self._rhand_geomids = set(physics.bind(self._rhand_geoms).element_id)
        self._head_geomids = set(physics.bind(self._head_geoms).element_id)
        self._hand_geomids = self._lhand_geomids | self._rhand_geomids

        self._arena.lever._specific_collision_geom_ids = (
            self._hand_geomids
        )  # pylint: disable=protected-access
        self._arena.spout._water._specific_collision_geom_ids = self._head_geomids

    def _update_state(self):
        # Enter TAP1 from BASE
        if self._state_logic == TwoTapState.BASE:
            if self._arena.lever.state == LeverState.TAP1:
                self._state_logic = TwoTapState.TAP1

        # Enter PASS or FAIL from TAP1
        elif self._state_logic == TwoTapState.TAP1:
            if self._arena.lever.state == LeverState.PASS:
                self._state_logic = TwoTapState.PASS
                self._arena.spout.release_droplet()
            elif self._arena.lever.state == LeverState.FAIL:
                 self._state_logic = TwoTapState.FAIL

        # Enter PASS_WATER_CONSUMED or BASE from PASS
        elif self._state_logic == TwoTapState.PASS:
            if self._arena.lever.state == LeverState.PASS:
                if not self._arena.spout.has_droplet():
                    self._state_logic = TwoTapState.PASS_WATER_CONSUMED
            elif self._arena.lever.state == LeverState.BASE:
                self._state_logic = TwoTapState.BASE

        # Enter BASE from PASS_WATER_CONSUMED
        elif self._state_logic == TwoTapState.PASS_WATER_CONSUMED:
            if self._arena.lever.state == LeverState.BASE:
                self._state_logic = TwoTapState.BASE

        # Enter BASE from FAIL
        elif self._state_logic == TwoTapState.FAIL:
            if self._arena.lever.state == LeverState.BASE:
                self._state_logic = TwoTapState.BASE


    def before_step(self, physics, action, random_state):
        super(TwoTap, self).before_step(physics, action, random_state)

    def after_step(self, physics, random_state):
        self._arena.lever.after_substep(physics, random_state)
        self.get_reward(physics)
        self._update_state()
        self._arena.spout._water._update_activation(physics)

    def should_terminate_episode(self, physics):
        failure_termination = False
        if failure_termination:
            self._discount = 0.0
            return True
        else:
            return False

    def get_reward(self, physics):
        reward = self._aliveness_reward

        # Calculate reward for closeness of hands to lever
        lhand_pos = physics.bind(self._lhand_body).xpos
        rhand_pos = physics.bind(self._rhand_body).xpos
        target_pos = physics.bind(self._arena.lever.geom).xpos
        lhand_rew = np.exp(-3.0 * sum(np.abs(lhand_pos - target_pos)))
        rhand_rew = np.exp(-3.0 * sum(np.abs(rhand_pos - target_pos)))
        closeness_reward = np.maximum(lhand_rew, rhand_rew)
        reward += 0.01 * closeness_reward * self._target_type_rewards[0]

        # Give a reward if first tap
        if self._state_logic == TwoTapState.BASE:
            if self._arena.lever.state == LeverState.TAP1:
                # self._state_logic = TwoTapState.TAP1
                reward += self._target_type_rewards[1]

        # Give a reward for second tap
        elif self._state_logic == TwoTapState.TAP1:
            if self._arena.lever.state == LeverState.PASS:
                # self._state_logic = TwoTapState.PASS
                # self._arena.spout.release_droplet()
                reward += self._target_type_rewards[2]

        # Give a reward for drinking the water
        elif self._state_logic == TwoTapState.PASS:
            if self._arena.lever.state == LeverState.PASS:
                if not self._arena.spout.has_droplet():
                    # self._state_logic = TwoTapState.PASS_WATER_CONSUMED
                    reward += self._target_type_rewards[3]
        return reward

    def get_discount(self, physics):
        del physics
        return self._discount
