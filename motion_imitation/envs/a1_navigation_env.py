from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import scipy
from absl import flags
from scipy import interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import time
import pybullet
import random

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

from mpc_controller import a1_sim as robot_sim
import gym
from gym import spaces
import math
import skfmm
from scipy import interpolate

FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300

_STANCE_DURATION_SECONDS = [
                               0.3
                           ] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 50

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)

a1_urdf = 'a1.urdf'


def wrap_angle(angle):
    """
    Taken from: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

NUM_LIDAR_RAYS = 100
LIDAR_RADIUS = 40

class A1NavigationEnv(gym.Env):
    # x, y, z. Used to normalize observations
    POSITION_RANGE_MIN = np.array([-5, -5, 0])
    POSITION_RANGE_MAX = np.array([5, 5, 2])
    GOAL_RADIUS = 0.25
    OBSTACLE_MARGIN = 0.25
    NORMALIZED_SPACE_MARGIN = 5.0
    def __init__(self, render=False,
                 record_video=False,
                 obs_type='pos_extended',
                 normalize_obs=True,
                 env_type='fixed',  # 'flat', 'fixed', 'random(Not Implemented)'
                 goal_type='origin',
                 goal_reward=1000., crash_penalty=100., fall_penalty=250.,
                 distance_signal_const=1.,
                 velocity=0.5, yaw_rate=4,
                 done_crash=True, done_fall=True,
                 get_constraint_cost=False,
                 constraint_cost_binary=False,
                 augment_worst_case=False,
                 num_action_hold=5,
                 num_obstacles_per_dim=6,
                 action_type='discrete', seed_num=10, max_time=30, print_step_log=False):
        super(A1NavigationEnv, self).__init__()

        self._is_render = render
        if render:
            self.connection_mode = pybullet.GUI
        else:
            self.connection_mode = pybullet.DIRECT

        # Main settings of the environment is done here
        self.env_type = env_type
        self.goal_type = goal_type
        # Max time for each episode
        self.MAX_TIME = max_time
        self._low_level_dt = 0.001
        self._num_action_hold = num_action_hold
        # NOTE: Physical timestep for one self.step() is
        # self._low_level_dt * ACTION_REPEAT(=5, in a1_sim.py) * self._num_action_hold
        self.distance_function = None
        self.terrain_info = None
        self.num_obstacles_per_dim = num_obstacles_per_dim

        self.record_video = record_video
        if self.record_video:
            self.p = bullet_client.BulletClient(
                connection_mode=self.connection_mode,
                options="--minGraphicsUpdateTimeMs=0 --mp4=\"test.mp4\" --mp4fps=" + str(24))
        else:
            self.p = bullet_client.BulletClient(
                connection_mode=self.connection_mode)
        self.seed(seed_num)
        self._setup_simulation()


        self.print_step_log = print_step_log

        self.vx = velocity
        self.yaw_rate = yaw_rate
        self.lin_speed_list = [0, velocity]
        self.ang_speed_list = [-yaw_rate, 0, yaw_rate]
        self.done_crash = done_crash
        self.done_fall = done_fall

        # Define reward related values
        self.reward = {}
        self.reward['goal'] = goal_reward
        self.reward['crash'] = -crash_penalty
        self.reward['fall'] = -fall_penalty
        self.reward['distance'] = distance_signal_const

        ## Defines observation space.
        # self.obs_function returns the unnormalized observation. Normalization is done as postprocess.
        self.normalize_obs = normalize_obs
        if obs_type == 'pos':
            self.obs_function = self._get_observation
            if self.normalize_obs:
                # for normalization.
                self.obs_scale = (self.POSITION_RANGE_MAX - self.POSITION_RANGE_MIN)
                self.obs_shift = self.POSITION_RANGE_MIN
                if goal_type == 'origin':
                    self.observation_space = spaces.Box(
                        0.0-self.NORMALIZED_SPACE_MARGIN, 1.0+self.NORMALIZED_SPACE_MARGIN, shape=(3,))
                elif goal_type == 'random':
                    self.obs_scale = np.concatenate([self.obs_scale, self.obs_scale[0:2]])
                    self.obs_shift = np.concatenate([self.obs_shift, self.obs_shift[0:2]])
                    self.observation_space = spaces.Box(
                        0.0-self.NORMALIZED_SPACE_MARGIN, 1.0+self.NORMALIZED_SPACE_MARGIN, shape=(5,))
            else:
                if goal_type == 'origin':
                    self.observation_space = spaces.Box(low=self.POSITION_RANGE_MIN, high=self.POSITION_RANGE_MAX)
                elif goal_type == 'random':
                    self.observation_space = spaces.Box(
                        low=np.concatenate([self.POSITION_RANGE_MIN, self.POSITION_RANGE_MIN[0:2]]),
                        high=np.concatenate([self.POSITION_RANGE_MAX, self.POSITION_RANGE_MAX[0:2]])
                    )
        elif obs_type == 'pos_extended':
            # pos_x, pos_y, pos_z, velocity, heading, yaw_rate, goal_state_x, goal_state_y
            self.obs_function = self._get_extended_observation
            RANGE_MAX = np.concatenate([self.POSITION_RANGE_MAX,
                                        3.0 * self.vx * np.ones(2),
                                        np.array([2.0, np.pi, 1.5 * yaw_rate])])
            RANGE_MIN = np.concatenate([self.POSITION_RANGE_MIN,
                                        -3.0 * self.vx * np.ones(2),
                                        -np.array([2.0, np.pi, 1.5 * yaw_rate])])
            if goal_type == 'random':
                RANGE_MAX = np.concatenate([RANGE_MAX, self.POSITION_RANGE_MAX[0:2]])
                RANGE_MIN = np.concatenate([RANGE_MIN, self.POSITION_RANGE_MIN[0:2]])
            if self.normalize_obs:
                self.obs_scale = (RANGE_MAX - RANGE_MIN)
                self.obs_shift = RANGE_MIN
                self.observation_space = spaces.Box(
                    0.0-self.NORMALIZED_SPACE_MARGIN, 1.0+self.NORMALIZED_SPACE_MARGIN, shape=RANGE_MAX.shape)
            else:
                self.observation_space = spaces.Box(low=RANGE_MIN, high=RANGE_MAX)
        elif obs_type == 'lidar':
            # self.obs_function = self._get_lidar_observation
            # obs_space = np.concatenate((np.array([-np.pi]), -LIDAR_RADIUS * np.ones(NUM_LIDAR_RAYS)))
            # self.observation_space = spaces.Box(low=obs_space, high=-obs_space)
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.observation_length = self.observation_space.shape[0]

        ## Defines Action Space:
        self.action_type = action_type
        if action_type == 'discrete':
            #   - Angle: -w_n, 0, +w_n
            #   - Velocity: 0, v
            self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(2)))
        elif action_type == 'discrete_serial':
            self.action_space = spaces.Discrete(6)
        else:
            raise NotImplementedError

        ## Safety constraint related (for safe RL algorithms, not used for normal RL)
        # If True, info in self.step returns the constraint.
        self.get_constraint_cost = get_constraint_cost
        if self.get_constraint_cost:
            self._cost = None
        # If True, return 1.0 under collision, 0.0 if not collided.
        # If False, return signed distance to the obstacle. (positive under collision, negative when collision-free).
        self.constraint_cost_binary = constraint_cost_binary
        self.augment_worst_case = augment_worst_case
        if self.augment_worst_case:
            self._worst_case_obs = None
            self._worst_case_cost = None
            self.observation_space = spaces.Box(
                0.0 - self.NORMALIZED_SPACE_MARGIN, 1.0 + self.NORMALIZED_SPACE_MARGIN, shape=(2*self.observation_length,))
            self.observation_length = self.observation_space.shape[0]

    def _setup_controller(self, robot):
        """Demonstrates how to create a locomotion controller."""
        desired_speed = (0, 0)
        desired_twisting_speed = 0

        gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            robot,
            stance_duration=_STANCE_DURATION_SECONDS,
            duty_factor=_DUTY_FACTOR,
            initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
            initial_leg_state=_INIT_LEG_STATE)
        state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                      window_size=20)
        sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_height=robot_sim.MPC_BODY_HEIGHT,
            foot_clearance=0.01)

        st_controller = torque_stance_leg_controller.TorqueStanceLegController(
            robot,
            gait_generator,
            state_estimator,
            desired_speed=desired_speed,
            desired_twisting_speed=desired_twisting_speed,
            desired_body_height=robot_sim.MPC_BODY_HEIGHT,
            body_mass=robot_sim.MPC_BODY_MASS,
            body_inertia=robot_sim.MPC_BODY_INERTIA)

        controller = locomotion_controller.LocomotionController(
            robot=robot,
            gait_generator=gait_generator,
            state_estimator=state_estimator,
            swing_leg_controller=sw_controller,
            stance_leg_controller=st_controller,
            clock=robot.GetTimeSinceReset)
        return controller

    def _update_controller_params(self, controller, lin_speed, ang_speed):
        controller.swing_leg_controller.desired_speed = lin_speed
        controller.swing_leg_controller.desired_twisting_speed = ang_speed
        controller.stance_leg_controller.desired_speed = lin_speed
        controller.stance_leg_controller.desired_twisting_speed = ang_speed

    def reset(self):
        if self.print_step_log:
            print('-----------------------reset------------------')
        self._setup_simulation()

        obs = self.obs_function()
        if self.normalize_obs:
            # Normalize observation.
            obs = (obs - self.obs_shift) / self.obs_scale
        if self.augment_worst_case:
            self._worst_case_obs = None
            self._worst_case_cost = None
            obs = np.concatenate([obs, obs])
        return obs

    def step(self, action_idx):
        self.current_time = self.robot.GetTimeSinceReset() - self.init_time
        # Updates the controller behavior parameters.
        if self.action_type == 'discrete':
            ang_speed_idx = action_idx[0]
            lin_speed_idx = action_idx[1]
        elif self.action_type == 'discrete_serial':
            ang_speed_idx = int(action_idx % 3)
            lin_speed_idx = int(action_idx / 3)
        lin_speed = [self._idx_to_lin_speed(lin_speed_idx), 0, 0]
        ang_speed = self._idx_to_ang_speed(ang_speed_idx)

        self._update_controller_params(self.controller, lin_speed, ang_speed)

        # Needed before every call to get_action().
        for _ in range(self._num_action_hold):
            self.controller.update()
            hybrid_action, info = self.controller.get_action()
            self.robot.Step(hybrid_action)

        if self._is_render:
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)

        obs = self.obs_function()
        if self.normalize_obs:
            # Normalize observation.
            obs = (obs - self.obs_shift) / self.obs_scale
        done, reach_goal, crash, fall = self._get_done()
        current_distance_to_goal = self.distance_to_goal()

        rew = 0
        if reach_goal:
            rew += self.reward['goal']
        else:
            # reward is based on change in distance to the goal.
            rew += self.reward['distance'] * (self.last_dist_to_goal - current_distance_to_goal)
            self.last_dist_to_goal = current_distance_to_goal
        if crash:       # Experiment 11/23: get rid of negative reward for crash.
            rew += self.reward['crash']
        if fall:
            rew += self.reward['fall']

        info = {}
        info['crash'] = crash
        info['fall'] = fall
        signed_distance = self.signed_distance()
        info['signed_distance'] = signed_distance
        info['distance_to_goal'] = current_distance_to_goal
        info['reach_goal'] = reach_goal
        if self.get_constraint_cost:
            info.update(self.cost(crash))

        if self.augment_worst_case and self.eval_worst_case(info):
            self._worst_case_obs = obs
            self._worst_case_cost = info['cost']
        if self.augment_worst_case:
            obs = np.concatenate([obs, self._worst_case_obs])
            info['worst_cost'] = self._worst_case_cost

        if self.print_step_log:
            print(f"\nCurrent time: {self.current_time}\n"
                  f"Goal State: {self.goal_state}\n"
                  f"Action: {action_idx}\n"
                  f"Signed Distance: {signed_distance}\n"
                  f"Dist. Goal: {current_distance_to_goal}\n"
                  f"Reward: {rew}\n"
                  f"Done: {done}")
            if self.get_constraint_cost:
                print(f"Constraint cost: {info['cost']}")
                if self.augment_worst_case:
                    print(f"Worst-case cost: {self._worst_case_cost}")

        return obs, rew, done, info

    def eval_worst_case(self, info):
        if self._worst_case_cost is None:
            return True
        if info['cost'] > self._worst_case_cost:
            return True
        return False

    def _setup_simulation(self):

        self.p.resetSimulation()

        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)

        self.p.setAdditionalSearchPath(pd.getDataPath())

        num_bullet_solver_iterations = 30

        self.p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)

        self.p.setPhysicsEngineParameter(enableConeFriction=0)
        self.p.setPhysicsEngineParameter(numSolverIterations=30)

        self.p.setTimeStep(self._low_level_dt)

        self.p.setGravity(0, 0, -9.8)
        self.p.setPhysicsEngineParameter(enableConeFriction=0)

        self.ground_id = self.p.loadURDF("plane.urdf")
        self.obstacles_id = None

        self._generate_terrain()

        idx_init = np.random.randint(0, self.x_init_safe.shape[0], 1)
        init_robot_xy_pos = [self.x_init_safe[idx_init], self.y_init_safe[idx_init]]

        init_robot_yaw = 2 * math.pi * (random.random() - 0.5)

        # Append z position.
        start_pos = init_robot_xy_pos + [robot_sim.START_POS[2]]
        start_orn = pybullet.getQuaternionFromEuler([0, 0, init_robot_yaw])

        self.robot_uid = self.p.loadURDF(robot_sim.URDF_NAME, start_pos, baseOrientation=start_orn)

        self.robot = robot_sim.SimpleRobot(self.p, self.robot_uid, simulation_time_step=self._low_level_dt)

        self.controller = self._setup_controller(self.robot)
        self.controller.reset()

        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)

        self.init_time = self.robot.GetTimeSinceReset()

        self.goal_state = self.generate_goal_state()
        self.last_dist_to_goal = self.distance_to_goal()

        return

    def generate_goal_state(self):
        # In case goal state is set throughout the training session.
        if self.goal_type == 'origin':
            init_goal_xy_pos = np.array([0, 0]).flatten()
        elif self.goal_type == 'random':
            idx_goal = np.random.randint(0, self.x_init_safe.shape[0], 1)
            init_goal_xy_pos = [self.x_init_safe[idx_goal], self.y_init_safe[idx_goal]]
        else:
            raise NotImplementedError

        goalVisualShape = self.p.createVisualShape(shapeType=self.p.GEOM_SPHERE,
                                                   radius=self.GOAL_RADIUS,
                                                   rgbaColor=[255, 0, 0, 1])
        self.p.createMultiBody(baseMass=0,
                               baseInertialFramePosition=[0, 0, 0],
                               baseVisualShapeIndex=goalVisualShape,
                               basePosition=[init_goal_xy_pos[0], init_goal_xy_pos[1], 0],
                               useMaximalCoordinates=True)
        return init_goal_xy_pos

    def _get_observation(self):
        pos = self.robot.GetTrueBasePosition()
        orn = wrap_angle(self.robot.GetBaseRollPitchYaw())

        obs = np.array([pos[0],  # x pos
                        pos[1],  # y pos
                        orn[2]]  # yaw
                       )

        if self.goal_type == 'random':
            # append goal x and goal y
            obs = np.concatenate([obs, self.goal_state])

        return obs

    def _get_extended_observation(self):

        pos = self.robot.GetTrueBasePosition()
        vel = self.robot.GetBaseVelocity()
        orn = wrap_angle(self.robot.GetBaseRollPitchYaw())
        orn_rate = self.robot.GetBaseRollPitchYawRate()

        obs = np.array([pos[0],   # x pos
                        pos[1],   # y pos
                        pos[2],   # z pos
                        vel[0],   # x vel
                        vel[1],   # y vel
                        vel[2],   # z vel
                        orn[2],   # yaw
                        orn_rate[2]]  # yaw_rate
                       )

        if self.goal_type == 'random':
            # append goal x and goal y
            obs = np.concatenate([obs, self.goal_state])

        return obs

    def _idx_to_ang_speed(self, idx):

        return self.ang_speed_list[idx]

    def _idx_to_lin_speed(self, idx):

        return self.lin_speed_list[idx]

    def distance_to_goal(self):
        """ Function that caculates the negative l2 norm distance to the goal state.
            neg_l2_dist = - math.sqrt(abs(x - x_g)^2 + abs(y - y_g)^2)
        """
        pos = self.robot.GetTrueBasePosition()
        s = np.array(pos[0:2])

        return math.sqrt(abs(s[0] - self.goal_state[0])**2 + abs(s[1] - self.goal_state[1])**2)

    def signed_distance(self):
        """ Computes the signed distance of a state to the obstacles.
        Args:
            s: State.
        Returns:
            Signed distance for the state s.
            positive if far from the obstacles.
            negative if intersecting with the obstacles.
        """

        pos = self.robot.GetTrueBasePosition()
        s = np.array(pos[0:2])

        return self.distance_function(s[0], s[1]).item()

    def _get_done(self):
        x, y, z = self.robot.GetTrueBasePosition()
        vel = self.robot.GetBaseVelocity()
        
        if self.obstacles_id is not None:
            contacts = self.p.getContactPoints(self.robot_uid, self.obstacles_id)
        else:
            contacts = []

        crash = len(contacts) > 0

        out_of_boundary = (x < self.POSITION_RANGE_MIN[0] or x > self.POSITION_RANGE_MAX[0]) or \
                          (y < self.POSITION_RANGE_MIN[1] or y > self.POSITION_RANGE_MAX[1]) or \
                          (vel[0] < - 4.0 * self.vx or vel[0] > 4.0 * self.vx) or \
                          (vel[1] < -4.0 * self.vx or vel[1] > 4.0 * self.vx)
        
        reach_goal = self.distance_to_goal() < self.GOAL_RADIUS
        
        fall = z < 0.2        


        done = out_of_boundary or reach_goal or (self.current_time > self.MAX_TIME)
        if self.done_crash:
            done = done or crash
        if self.done_fall:
            done = done or fall

        if crash:
            print("Robot crashed into an obstacle!")

        if reach_goal:
            print("Robot reached its goal state!")

        if fall:    # If the robot falls down, there's no way it's getting up anyway.
            print("Robot fell down.")

        if done:
            print("current time = {}".format(self.current_time))

        return done, reach_goal, crash, fall

    def cost(self, crash):
        cost = {}
        if self.constraint_cost_binary:
            cost['cost_pillars'] = 1.0 if crash else 0.0
        else:
            cost['cost_pillars'] = - self.signed_distance()

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))
        self._cost = cost
        return cost

    def _get_lidar_observation(self):
        # JC: This function is not reviewed yet.
        raise NotImplementedError

        robot_pos = list(self.robot.GetTrueBasePosition())
        robot_pos[2] = robot_pos[2] + 0.2

        yaw = wrap_angle(self.robot.GetBaseRollPitchYaw())[2]
        radius = 20
        num_rays = 100
        ray_deltas = np.array(
            [[radius * np.cos(th), radius * np.sin(th), 0] for th in np.linspace(-np.pi/2 + yaw, np.pi/2 + yaw, num_rays)])

        ray_from = np.tile(np.array(robot_pos), (num_rays, 1))

        ray_to = ray_from + ray_deltas

        rays = self.p.rayTestBatch(ray_from, ray_to)

        # for idx in range(num_rays):
        #     self.p.addUserDebugLine(ray_from[idx, :], ray_to[idx, :], lineColorRGB=[1, 0, 0],
        #                             lineWidth=1,
        #                             lifeTime=2)

        return np.concatenate((np.array([yaw]), np.array([rays[idx][2] for idx in range(num_rays)])))

    def _generate_terrain(self):
        side_augment = 0.2
        num_x = int((self.POSITION_RANGE_MAX[0] - self.POSITION_RANGE_MIN[0] + 2 * side_augment) * 20)
        num_y = int((self.POSITION_RANGE_MAX[1] - self.POSITION_RANGE_MIN[1] + 2 * side_augment) * 20)
        dx = (self.POSITION_RANGE_MAX[0] - self.POSITION_RANGE_MIN[0] + 2 * side_augment) / num_x
        dy = (self.POSITION_RANGE_MAX[1] - self.POSITION_RANGE_MIN[1] + 2 * side_augment) / num_y

        if self.env_type == 'random' or self.terrain_info is None:
            x = np.linspace(self.POSITION_RANGE_MIN[0] - side_augment, self.POSITION_RANGE_MAX[0] + side_augment, num_x)
            y = np.linspace(self.POSITION_RANGE_MIN[1] - side_augment, self.POSITION_RANGE_MAX[1] + side_augment, num_y)
            [X, Y] = np.meshgrid(x, y)
            Z = -1 * np.ones(X.shape)

            if self.env_type != 'unbounded':
                # Create Boundary
                Z[X < self.POSITION_RANGE_MIN[0]] = 1
                Z[X > self.POSITION_RANGE_MAX[0]] = 1
                Z[Y < self.POSITION_RANGE_MIN[1]] = 1
                Z[Y > self.POSITION_RANGE_MAX[1]] = 1

            # Creat pillars
            if self.env_type == 'fixed':
                num_pillar_x = self.num_obstacles_per_dim
                num_pillar_y = self.num_obstacles_per_dim
                pillar_interval_x = (self.POSITION_RANGE_MAX[0] - self.POSITION_RANGE_MIN[0]) / num_pillar_x
                pillar_interval_y = (self.POSITION_RANGE_MAX[1] - self.POSITION_RANGE_MIN[1]) / num_pillar_y
                if num_pillar_x < 5:
                    pillar_width = 0.5
                else:
                    pillar_width = 0.3

                for i_x in range(num_pillar_x):
                    for i_y in range(num_pillar_y):
                        pillar_x_start = self.POSITION_RANGE_MIN[0] + (0.5 + i_x) * pillar_interval_x - 0.5 * pillar_width
                        pillar_x_end = self.POSITION_RANGE_MIN[0] + (0.5 + i_x) * pillar_interval_x + 0.5 * pillar_width
                        pillar_y_start = self.POSITION_RANGE_MIN[1] + (0.5 + i_y) * pillar_interval_y - 0.5 * pillar_width
                        pillar_y_end = self.POSITION_RANGE_MIN[1] + (0.5 + i_y) * pillar_interval_y + 0.5 * pillar_width
                        a1 = X >= pillar_x_start
                        a2 = X <= pillar_x_end
                        a3 = Y >= pillar_y_start
                        a4 = Y <= pillar_y_end
                        a = a1 * a2 * a3 * a4
                        Z[a] = 1

            d = skfmm.distance(-Z, dx=[dx, dy])
            d = d - self.OBSTACLE_MARGIN
            # Update distance function and terrain info
            self.distance_function = interpolate.interp2d(x, y, d, kind='cubic')
            self.terrain_info = {}
            self.terrain_info['X'] = X
            self.terrain_info['Y'] = Y
            self.terrain_info['Z'] = Z
            self.terrain_info['d'] = d

        # Set up safe initial state table to sample initial state and goal state.
        init_state_margin = 0.05
        idx_init_safe = self.terrain_info['d'] > init_state_margin
        self.x_init_safe = self.terrain_info['X'][idx_init_safe]
        self.y_init_safe = self.terrain_info['Y'][idx_init_safe]

        if self.env_type != 'unbounded':
            heightfieldData = np.reshape(3 * self.terrain_info['Z'], (num_x * num_y))
            p = self.p
            terrainShape = self.p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                                  meshScale=[dx, dy, 0.5],
                                                  heightfieldTextureScaling=(num_x - 1) / 2,
                                                  heightfieldData=heightfieldData,
                                                  numHeightfieldRows=num_x,
                                                  numHeightfieldColumns=num_y)

            self.obstacles_id = p.createMultiBody(0, terrainShape)
            p.resetBasePositionAndOrientation(self.obstacles_id, [0, 0, 0], [0, 0, 0, 1])
            p.changeVisualShape(self.obstacles_id, -1, rgbaColor=[1, 1, 1, 1])

        return
