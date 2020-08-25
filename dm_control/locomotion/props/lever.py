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
"""A non-colliding sphere that is pressed through touch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control import mjcf
import numpy as np
import enum

BASE_COLOR = (0.5, 0.5, 0.5, 1)
ACTIVATED_COLOR = (0.0, 0.0, 0.5, 1)
TAP1_COLOR = (0.0, 0.0, 0.5, 1)
PASS_COLOR = (0.0, 0.5, 0.0, 1)
FAIL_COLOR = (0.5, 0, 0.0, 1)

class LeverState(enum.IntEnum):
    BASE = 0
    TAP1 = 1
    PASS = 2  # Passed, water available
    FAIL = 3


class Lever(composer.Entity):
    """A non-colliding lever that is pressed when pressed.

  The target indicates if it has been pressed at least once and pressed at least
  twice this episode with a two-bit pressed state tuple.
  """

    def _build(
        self,
        pos=[0, -0.006, 0],
        size=[0.007, 0.012, 0.001],
        rgb1=(0.5, 0.5, 0.5, 1),
        rgb2=(0.2, 0.2, 0.2,),
        specific_collision_geom_ids=None,
        name="lever",
        trigger_position=np.pi / 4,
    ):
        self._mjcf_root = mjcf.RootElement(model=name)
        self._lever = self.mjcf_model.worldbody.add("body", name=name, pos=pos)
        self._lever_hinge = self._lever.add(
            "joint",
            name="lever_hinge",
            type="hinge",
            axis=[1, 0, 0],
            limited=True,
            armature=1e-04,
            damping=0.001,
            stiffness=0.02,
            springref=0,
            range=[0, np.pi / 4],
        )
        self._geom = self._lever.add(
            "geom",
            name=name,
            type="box",
            size=size,
            pos=pos,
            rgba=rgb1,
            friction=[0.7, 0.005, 0.0001],
            solref=[0.005, 1],
            solimp=[0.99, 0.9999, 0.001],
        )
        self._geom_id = -1
        self._pressed = False
        self._triggered = False
        self._trigger_time = 0.0
        self._specific_collision_geom_ids = specific_collision_geom_ids
        self._trigger_position = trigger_position

    @property
    def geom(self):
        return self._geom

    @property
    def pressed(self):
        return self._pressed

    @property
    def triggered(self):
        return self._triggered

    def reset(self, physics):
        self._pressed = False
        physics.bind(self._geom).rgba = BASE_COLOR

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def initialize_episode_mjcf(self, unused_random_state):
        self._pressed = False
        self._triggered = False

    def _update_press_and_trigger(self, physics):
        # Update pressed state
        lever_position = np.copy(physics.named.data.qpos["lever/lever_hinge"])
        already_pressed = self._pressed
        self._pressed = self._trigger_position < lever_position
        self._triggered = self._pressed and not already_pressed

        # Save the trigger time
        if self._triggered:
            self._trigger_time = physics.time()

    def _update_activation(self, physics):
        self._update_press_and_trigger(physics)

        # If it is pressed, change the color.
        if self._pressed:
            physics.bind(self._geom).rgba = ACTIVATED_COLOR
        return

    def initialize_episode(self, physics, unused_random_state):
        self._geom_id = physics.model.name2id(self._geom.full_identifier, "geom")
        self._update_activation(physics)

    def after_substep(self, physics, unused_random_state):
        self._update_activation(physics)


class TwoTapLever(Lever):
    """A colliding sphere that is pressed through touch.

  The target indicates if it has been touched at least once and touched at least
  twice this episode with a two-bit pressed state tuple.  It remains pressed
  for the remainder of the current episode.

  The target is automatically reset at episode initialization.
  """

    def _build(
        self,
        pos=[0, -0.006, 0],
        size=[0.007, 0.012, 0.001],
        rgb_base=BASE_COLOR,
        rgb_tap1=TAP1_COLOR,
        rgb_pass=PASS_COLOR,
        rgb_fail=FAIL_COLOR,
        specific_collision_geom_ids=None,
        name="lever",
        trigger_position=np.pi / 4,
        tap_interval=[0.6, 0.8],
        penalty_interval=2,
        inter_trial_interval=1.2,
    ):
        """Builds this target sphere.

    Args:
      radius: The radius (in meters) of this target sphere.
      height_above_ground: The height (in meters) of this target above ground.
      rgb_initial: A tuple of two colors for the stripe pattern of the target.
      rgb_interval: A tuple of two colors for the stripe pattern of the target.
      rgb_final: A tuple of two colors for the stripe pattern of the target.
      touch_debounce: duration to not count second touch.
      specific_collision_geom_ids: Only activate if collides with these geoms.
      name: The name of this entity.
    """
        super(TwoTapLever, self)._build(
            pos=pos,
            size=size,
            specific_collision_geom_ids=specific_collision_geom_ids,
            name=name,
            trigger_position=trigger_position,
        )

        self._state = LeverState.BASE
        self.tap_interval = tap_interval
        self.penalty_interval = penalty_interval
        self.inter_trial_interval = inter_trial_interval
        self._specific_collision_geom_ids = specific_collision_geom_ids
        self._previous_trigger_time = 0.0

    @property
    def state(self):
        """Whether this target has been reached during this episode."""
        return self._state

    def reset(self, physics):
        super(TwoTapLever, self).reset(physics)
        self._state = LeverState.BASE
        self._previous_trigger_time = 0.0
        self._trigger_time = 0.0
        self._pressed = False

    def initialize_episode_mjcf(self, unused_random_state):
        super(TwoTapLever, self).initialize_episode_mjcf(unused_random_state)
        self._state = LeverState.BASE
        self._previous_trigger_time = 0.0


    def _update_activation(self, physics):
        super(TwoTapLever, self)._update_press_and_trigger(physics)
        interval = physics.time() - self._trigger_time
        if self._triggered:
            interval = self._trigger_time - self._previous_trigger_time
            self._previous_trigger_time = self._trigger_time

        # Start first tap
        if self._state == LeverState.BASE and self._triggered:
            self._state = LeverState.TAP1
            physics.bind(self._geom).rgba = TAP1_COLOR

        # If too long, fail
        elif (self._state == LeverState.TAP1) and (interval > self.tap_interval[1]):
            self._state = LeverState.FAIL
            physics.bind(self._geom).rgba = FAIL_COLOR

        # Handle Second tap
        elif self._state == LeverState.TAP1 and self._triggered:
            # If the correct interval, pass, otherwise fail
            if interval > self.tap_interval[0] and interval < self.tap_interval[1]:
                self._state = LeverState.PASS
                physics.bind(self._geom).rgba = PASS_COLOR
            else:
                self._state = LeverState.FAIL
                physics.bind(self._geom).rgba = FAIL_COLOR

        # Handle penalty interval
        elif self._state == LeverState.FAIL:
            if interval > self.penalty_interval:
                self._state = LeverState.BASE
                physics.bind(self._geom).rgba = BASE_COLOR

        # Handle inter trial interval
        elif self._state == LeverState.PASS:
            if interval > self.inter_trial_interval:
                self._state = LeverState.BASE
                physics.bind(self._geom).rgba = BASE_COLOR
