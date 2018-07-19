import numpy as np
from sandbox.embed2learn.envs.mujoco.sawyer_pick_and_place import PickAndPlaceEnv

env = PickAndPlaceEnv()
env.reset()
action = np.array([0.3, 0.2, -0.05])


for i in range(4000):
    env.render()
    env.sim.data.set_mocap_pos('mocap', action)
    env.sim.data.set_mocap_quat('mocap', np.array([0, 1, 1, 0]))
    env.sim.step()
print(action, env.sim.data.mocap_pos, env.sim.data.get_site_xpos('grip'))
action = np.array([0, 0, 0, 10])
for i in range(1000):
    env.render()
    env.step(action)
print(action, env.sim.data.mocap_pos, env.sim.data.get_site_xpos('grip'))


action = np.array([0, 0, -0.5, 10])
for i in range(1000):
    env.render()
    env.step(action)
print(action, env.sim.data.mocap_pos, env.sim.data.get_site_xpos('grip'))

action = np.array([0, 0, 0, -10])
for i in range(4000):
    env.render()
    env.step(action)
print(action, env.sim.data.mocap_pos, env.sim.data.get_site_xpos('grip'))

action = np.array([0, 0, 0.5, -10])
for i in range(4000):
    env.render()
    env.step(action)
print(action, env.sim.data.mocap_pos, env.sim.data.get_site_xpos('grip'))
