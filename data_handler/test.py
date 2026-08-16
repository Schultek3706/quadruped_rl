import mujoco
import numpy as np
import mujoco.viewer
from pathlib import Path
from Sim_data import DataHandler
import time
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    project_root = here.parent
    xml = project_root / "Robot" / "robot.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    seed = 42
    rng = np.random.default_rng(42)

    sign = np.ones(12)
    handler = DataHandler(model,seed = seed)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(10):
            mujoco.mj_resetData(model, data)
            handler.reset_world_random(model, data,i / 10)
            phase = rng.uniform(-1, 1, 12)
            for j in range(10000):
                time.sleep(0.001)
                sign = np.where(phase >= 1,-1, sign)
                sign = np.where(phase <= 0,1, sign)
                phase += sign/1000
                phase.clip(-1,1)
                pos_goal = handler.set_servos(phase)
                if j%1000 == 0:
                    print(handler.step(data,s = (j * i) / 10000))
                data.ctrl[:] = pos_goal
                mujoco.mj_step(model,data,1)
                viewer.sync()
        pass