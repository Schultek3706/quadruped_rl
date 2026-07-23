<h1>quadruped-rl</h1>
<h2>Goal</h2>
<p>
The goal of this proejct is to build a quadruped robot for a simulated environment and training it to walk by using reinforcement learning.
In the process i will log my progress and various training metrics for an analysis of the training and the end-result.
</p>
<h2>The Robot</h2>
<p>
The Robot is a 12 DOF quadruped robot which is structured similiar to spiders. For the simulation environment MuJoCo will be used due to it being used for similar locomotion
simulation and its easier handling compared to other ones. To simulate the robot we first need a digital version of it as Mujoco works with mjcf files which are written in xml.
</p>
