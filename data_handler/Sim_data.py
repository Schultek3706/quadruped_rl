import mujoco
import torch
import numpy as np
class DataHandler:
    def __init__(self,nominal,noise_ratio):
        noise_ratio = {"servo_pos_noise":0.01, "servo_vel_noise":0.08, "servo_pos_bias": 0.09, "servo_vel_bias":0.05,
                       "base_gyro_noise":0.05, "base_gyro_bias":0.05, "sim_imu_noise":0.04}
        self.servo_pos_nr = noise_ratio["servo_pos_noise"]
        self.servo_vel_nr = noise_ratio["servo_vel_noise"]
        self.servo_pos_bias = noise_ratio["servo_pos_bias"]
        self.servo_vel_bias = noise_ratio["servo_vel_bias"]
        self.gyro_nr = noise_ratio["base_gyro_noise"]
        self.gyro_bias = noise_ratio["base_gyro_bias"]
        self.sim_imu_nr = noise_ratio["sim_imu_noise"]
        self.nominal = nominal
        self.noise_ratio = noise_ratio
        self.servo_pos_bias_episode = None
        self.servo_vel_bias_episode = None
        self.accel_bias_episode = None
        self.gyro_bias_episode = None
        pass
    def step(self, data, s):
        servo_pos = torch.cat((data.sensor("hip_1_pos").data, data.sensor("knee_1_pos").data,data.sensor("ankle_1_pos").data,
                               data.sensor("hip_2_pos").data, data.sensor("knee_2_pos").data,data.sensor("ankle_2_pos").data,
                               data.sensor("hip_3_pos").data, data.sensor("knee_3_pos").data,data.sensor("ankle_3_pos").data,
                               data.sensor("hip_4_pos").data, data.sensor("knee_4_pos").data,data.sensor("ankle_4_pos").data,),dim=0)
        for i in range(len(servo_pos)):
            servo_pos[i] += s*(np.random.uniform(-self.servo_pos_nr,self.servo_pos_nr) + self.servo_pos_bias_episode)

        servo_vel = torch.cat((data.sensor("hip_1_vel").data, data.sensor("knee_1_vel").data,data.sensor("ankle_1_vel").data,
                               data.sensor("hip_2_vel").data, data.sensor("knee_2_vel").data,data.sensor("ankle_2_vel").data,
                               data.sensor("hip_3_vel").data, data.sensor("knee_3_vel").data,data.sensor("ankle_3_vel").data,
                               data.sensor("hip_4_vel").data, data.sensor("knee_4_vel").data,data.sensor("ankle_4_vel").data,),dim=0)

        for i in range(len(servo_vel)):
            servo_vel[i] *= (1+s*(np.random.uniform(-self.servo_vel_nr,self.servo_vel_nr + self.servo_vel_bias_episode)))

        quat_data = data.sensor("imu_quat").data
        sim_imu_reading = self.quat_to_gravity(quat_data)

        for i in range(len(sim_imu_reading)):
            sim_imu_reading[i] += s*np.random.uniform(-self.sim_imu_nr,self.sim_imu_nr)

        gyro_data = data.sensor("imu_gyro").data
        for i in range(len(gyro_data)):
            gyro_data[i] += s*(np.random.uniform(-self.gyro_nr,self.gyro_nr) + self.gyro_bias_episode)
        return np.concat((servo_pos, servo_vel, sim_imu_reading, gyro_data), axis=0)
    def set_servos(self,action,s,data):
        for i in range(0,len(action),3):
            action[i] = self.denorm_hip(action[i])
            action[i + 1] = self.denorm_knee(action[i + 1])
            action[i + 2] = self.denorm_ankle(action[i + 2])
        pass
    def denorm_hip(self,norm_val):
        return norm_val*0.69
    def denorm_knee(self,norm_val):
        return norm_val*1.5
    def denorm_ankle(self,norm_val):
        return norm_val*0.845
    def reset_noise(self,s):
        self.servo_pos_bias_episode = s*np.random.uniform(-self.servo_pos_bias,self.servo_pos_bias)
        self.servo_vel_bias_episode = s*np.random.uniform(-self.servo_vel_bias,self.servo_vel_bias)
        self.gyro_bias_episode = s*np.random.uniform(-self.gyro_bias,self.gyro_bias)

        pass
    def quat_to_gravity(self,quat, gravity_mag=9.81):
        # quat = [w, x, y, z]
        w, x, y, z = quat

        # Global gravity vector (assuming Z is up, gravity points down)
        # If your simulation uses Y-up, adjust accordingly (e.g., [0, -g, 0])
        g_global = np.array([0, 0, -gravity_mag])

        # Rotation matrix derived from quaternion to rotate global vector to local frame
        # This is equivalent to R^T * g_global where R is the rotation matrix of q
        R = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
        ])

        # For "sensor frame" gravity, we often want the vector as measured by the sensor
        # which is effectively the global gravity rotated by the inverse of the body orientation.
        # Since R rotates local -> global, R.T rotates global -> local.
        g_local = R.T @ g_global

        return g_local