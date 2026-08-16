import mujoco
import torch
import numpy as np
joints = ["hip_1","knee_1","ankle_1",
          "hip_2","knee_2","ankle_2",
          "hip_3","knee_3","ankle_3",
          "hip_4","knee_4","ankle_4",]
noise_ratio = {"servo_pos": 0.005, "servo_vel": 0.05,"servo_pos_bias": 0.01, "servo_vel_bias": 0.05,
                "base_gyro": 0.05, "base_gyro_bias": 0.02, "sim_imu": 0.02,"zero_offset":0.01}
world_noise = {"base_mass":0.15, "base_pos":0.015, "infill_mass": 0.1, "gearbox": 0.3, "armature": 0.3,"friction": 0.05,"strength":0.15}
class DataHandler:
    def __init__(self,model,seed=42):
        self.n_joints = len(joints)
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        joint_ids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,j) for j in joints]
        self.joint_dof_adr = np.array([model.jnt_dofadr[i] for i in joint_ids])

        sensor_id_pos = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SENSOR,j + "_pos") for j in joints]
        self.sensor_pos_adr = np.array([model.sensor_adr[i] for i in sensor_id_pos])

        sensor_id_vel = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, j + "_vel") for j in joints]
        self.sensor_vel_adr = np.array([model.sensor_adr[i] for i in sensor_id_vel])

        quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        quat_ad = model.sensor_adr[quat_id]
        self.quat_adr = np.arange(quat_ad, quat_ad + 4)
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        gyro_ad = model.sensor_adr[gyro_id]
        self.gyro_adr = np.arange(gyro_ad, gyro_ad + 3)

        self.num_ac = model.nu
        act_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
        try:
            self.ctrl_idx = np.array([act_of[i] for i in joint_ids])
        except KeyError as e:
            raise KeyError(e)

        cr = model.actuator_ctrlrange[self.ctrl_idx]
        self.ctrl_lo, self.ctrl_hi = cr[:, 0].copy(), cr[:, 1].copy()
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        self.nominal = {
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "body_ipos": model.body_ipos.copy(),
            "dof_damping": model.dof_damping.copy(),
            "dof_armature": model.dof_armature.copy(),
            "dof_frictionloss": model.dof_frictionloss.copy(),
            "floor_friction": model.geom_friction[self.floor_id, 0].copy(),
            "forcerange": model.actuator_forcerange.copy(),
        }
        self.noise_ratio = noise_ratio
        self.world_noise = world_noise
        self.rng = np.random.default_rng(seed=seed)
        self.pos_bias_ep = None
        self.servo_vel_bias_ep = None
        self.gyro_bias_ep = None
        self.zero_offset_ep = None
        self.reset_noise(0.0)
        pass
    def step(self, data, s):
        sd = data.sensordata.copy()

        servo_pos = sd[self.sensor_pos_adr] + self.pos_bias_ep
        servo_pos += s*self.rng.normal(0,self.noise_ratio["servo_pos"],size=self.n_joints)

        servo_vel = sd[self.sensor_vel_adr]
        servo_vel += s*self.rng.normal(0,self.noise_ratio["servo_vel"],size=self.n_joints)

        grav = self.projected_gravity(sd[self.quat_adr])
        grav += s*self.rng.normal(0,self.noise_ratio["sim_imu"],size=3)

        gyro = sd[self.gyro_adr]
        gyro += s*self.rng.normal(0,self.noise_ratio["base_gyro"],size=3)
        return np.concat((servo_pos, servo_vel, grav, gyro))
    @staticmethod
    def projected_gravity(quat):
        conj = np.zeros(4)
        mujoco.mju_negQuat(conj, quat.astype(np.float64))
        down = np.zeros(3)
        mujoco.mju_rotVecQuat(down, np.array([0.0, 0.0, -1.0]), conj)
        return down

    def set_servos(self,action):
        target = self.ctrl_half * action + self.zero_offset_ep
        target = np.clip(target, self.ctrl_lo, self.ctrl_hi)

        ctrl = np.zeros(self.n_joints)
        ctrl[self.ctrl_idx] = target
        return ctrl

    def denorm_hip(self,norm_val):
        return norm_val*0.69
    def denorm_knee(self,norm_val):
        return norm_val*1.5
    def denorm_ankle(self,norm_val):
        return norm_val*0.845
    def reset_noise(self,s):
        s = float(np.clip(s,0,1))
        self.pos_bias_ep = s*self.rng.uniform(-self.noise_ratio["servo_pos_bias"],self.noise_ratio["servo_pos_bias"],size=self.n_joints)
        self.gyro_bias_ep = s*self.rng.uniform(-self.noise_ratio["base_gyro_bias"],self.noise_ratio["base_gyro_bias"],size=3)
        self.zero_offset_ep = s*self.rng.uniform(-self.noise_ratio["zero_offset"],self.noise_ratio["zero_offset"],size=self.n_joints)
        pass

    def reset_world_random(self,model,data,s):
        s = float(np.clip(s, 0.0, 1.0))

        f = 1.0+s*self.rng.uniform(self.world_noise["infill_mass"],self.world_noise["infill_mass"],model.nbody)
        f[0] = 0.0
        f[self.base_id] = 1.0+s*self.rng.uniform(self.world_noise["base_mass"],self.world_noise["base_mass"])
        model.body_mass[:] = self.nominal["body_mass"] * f
        model.body_inertia[:] = self.nominal["body_inertia"] * f[:, None]

        model.body_ipos[:] = self.nominal["body_ipos"]
        model.body_ipos[self.base_id] += s*self.rng.uniform(-self.world_noise["base_pos"],self.world_noise["base_pos"],size=3)

        da = self.joint_dof_adr
        model.dof_damping[da] = self.nominal["dof_damping"][da] * (1 + s*self.rng.uniform(-self.world_noise["gearbox"],self.world_noise["gearbox"],size=self.n_joints))
        model.dof_armature[da] = self.nominal["dof_armature"][da] * (1 + s*self.rng.uniform(-self.world_noise["armature"],self.world_noise["armature"],size=self.n_joints))
        model.dof_frictionloss[da] = (self.nominal["dof_frictionloss"][da] + s*self.rng.uniform(0,self.world_noise["friction"],size=self.n_joints))

        ac_weakness = 1.0 + s*self.rng.uniform(-self.world_noise["strength"],0,size=(self.num_ac,1))
        model.actuator_forcerange[:] = self.nominal["forcerange"]*ac_weakness

        model.geom_priority[self.floor_id] = 1
        model.geom_friction[self.floor_id,0] = (self.nominal["floor_friction"]*(1+s*self.rng.uniform(-self.world_noise["friction"],self.world_noise["friction"])))

        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)
        self.reset_noise(s)

        return model,data