from data_handler.Sim_data import DataHandler
import gymnasium as gym
import numpy as np
from pathlib import Path
import mujoco

class RunClass(gym.Env):
    def __init__(self, seed, robot_path):
        super().__init__()
        here = Path(__file__).resolve().parent
        project_root = here.parent
        xml = project_root / "Robot" / robot_path
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        frames_stacked = 20
        self.observation_space = gym.spaces.Box(low = -10*np.ones((30*frames_stacked,)),high = 10*np.ones((30*frames_stacked,)),shape = (30*frames_stacked,))
        self.obs = np.ones(30*frames_stacked)
        self.action_space = gym.spaces.Box(low=-1*np.ones(12),high=1*np.ones(12),shape = (12,))
        self.orientation = np.ones(50*5) #5 seconds
        self.orientation = self.orientation.astype(bool)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.handler = DataHandler(self.model,self.seed)
        self.ep_c = 0
        self.ep_total = 10000
        self.episode_step = 0
        self.reset()
    def reset(self, seed=None, options = None):
        s = self.ep_c / self.ep_total
        super().reset()
        self.model, self.data = self.handler.reset_world_random(self.model,self.data,s)
        self.ep_c += 1
        for i in range(20):
            servo_pos = self.data.sensordata[self.handler.sensor_pos_adr].copy()
            goal_pos = servo_pos + s*self.rng.uniform(-0.05,0.05,servo_pos.shape)
            self.data.ctrl[:] = goal_pos
            mujoco.mj_step(self.model, self.data, 10)
            obs = self.handler.get_obs(self.data,s)
            self.obs = np.append(self.obs, obs)
            self.obs = np.delete(self.obs, np.s_[0:30])
        reward_data = self.handler.reward_data(self.data)
        info = {"speed": reward_data[0][0], "gravity_down": reward_data[1][2]}
        return self.obs.copy(), info
    def step(self, action): # 50hz control loop
        s = self.ep_c/self.ep_total
        s = np.clip(s,0,1)

        servo_goal = self.handler.set_servos(action)
        self.data.ctrl[:] = servo_goal
        mujoco.mj_step(self.model,self.data,10)
        obs_data = self.handler.get_obs(self.data,s)
        reward_data = self.handler.reward_data(self.data)
        if reward_data[1][2] < 0:
            self.orientation = np.append(self.orientation,True)
        else:
            self.orientation = np.append(self.orientation,False)
        self.orientation = np.delete(self.orientation,0)
        if np.all(self.orientation == False):
            terminated = True
        else:
            terminated = False
        if self.episode_step >= 50*60:
            truncated = True
        else:
            truncated = False
        reward = self.reward(reward_data,action,s)
        self.obs = np.append(self.obs,obs_data)
        self.obs = np.delete(self.obs,np.s_[0:30])
        info = {"speed": reward_data[0][0],"gravity_down": reward_data[1][2]}
        return self.obs.copy(), reward, terminated, truncated, info
    def reward(self,reward_data,action,s):
        v_base, grav, height, servo_pos, servo_vel, rot_vel, ac_force = reward_data
        vel_reward = (5 + s*5)*v_base[0]
        vel_penalty = -(1 + s)*v_base[1]
        pen_grav = -(1-0.5*s)*np.sqrt(grav[0]**2 + grav[1]**2)
        if grav[2] > 0:
            pen_grav -= 2.5
        pen_height = -0.1*(1-s)*np.exp(-height*10)
        pen_servo_pos = -0.1*(2-s)*(np.exp(np.sum(np.abs(servo_pos-action[:-1])))-1)
        pen_servo_vel = -0.025*(2-s)*(np.exp(np.sum(np.abs(servo_vel)))-1)
        pen_rot_vel = -0.25*(2-s)*(np.exp(np.sum(np.abs(rot_vel)))-1)
        pen_ac_force = -0.5*np.sum(np.abs(ac_force))/100
        total_reward = vel_reward + vel_penalty + pen_grav + pen_height + pen_servo_pos + pen_servo_vel + pen_rot_vel + pen_ac_force
        return total_reward