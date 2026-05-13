<h1>quadruped-rl</h1>
<h2>Goal</h2>
<p>
In this project a quadruped robot is learning to walk in sim.Furthermore, the training performance, <br>
learned generalized behavior and how it differs based on changes in the parameters will be analyzed.
</p>
<h2>Setup</h2>
<p>
The Robot is a 12 DOF quadruped robot structured similiar to anthropods. The chosen simulation engine is Mujoco for <br>
its many uses in similar projects and great physics simulation capabilities. For better general behavior the terrain, <br>
friction, inertias and other parameters will be randomized through training as well as noise injection in the sensor data <br>
for a sim to real deployment possibility. For the reinforcement learning part Mujoco is wrapped in a gym environment which <br>
interacts with an MLP through sb3's PPO implementation.
</p>
<h2>The Simulated Robot</h2>
<p>
To create an easy adjustable robot in the mjcf format for Mujoco a generation script is more optimal than a direct model <br>
due to its size and need adjust multiple repetitive parts each by hand.</p>