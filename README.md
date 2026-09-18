<h1>quadruped-rl</h1>
<h2>Goal</h2>
<p>
The goal of this proejct is to build a quadruped robot for a simulated environment and training it to walk by using reinforcement learning.
In the process i will log my progress and various training metrics for an analysis of the training and the end-result.
</p>
<h2>The Robot</h2>
<p>
The Robot is a 12 DOF quadruped robot which is structured similiar to spiders. For the physics engine MuJoCo will be used due to it being used for similar projects 
and its easier handling compared to other ones. To simulate the robot we first need a digital version of it as Mujoco works with mjcf files which are written in xml.
</p>
<h2>The robots specifications</h2>
<p>
The base is: 8cm x 8cm x 2cm<br>
Mass of the base body: 550g<br>
The legs are placed or the diagonals of the bottom of the base and extend from there<br>
The upper legs are: 7.5cm long<br>
The lower legs are: 9.5cm long<br>
The feet are: 12.5cm long<br>
Density is 1,25g/cm^3 for all legs and feet which around equal to PLA<br>
The Servos for movement are also simulated by using a box weighing 60g<br>
</p>
<h2>The Randomization in observations and world scene</h2>
<p>
The following parts change with a variation per episode to simulate different settings:<br>
The mass of the base with 15%<br>
The center of mass of the base 1.5mm<br>
The mass of the remaining bodies with 10%<br>
The friction of the servo gearbox with 30%<br>
The reflected inertia of servo rotors with 30%<br>
The friction from wear down additive 5% <br>
The floor friction 40% with 100% friction priority to the floor<br>
Strength degradation up to 15%<br>
Zero-Offset of the servos up 0.01 rad<br>
</p>
<p>
In each observation for the model later we add noise to each component accordingly: (all additive)<br>
Servo-position have 15%<br>
Servo-velocity 5%<br>
Servo-position bias up to 1%<br>
The gyro of the base 5%<br>
The base-gyro bias 2%<br>
The simulated imu 2%<br>
</p>
<h2>The Reward function</h2>
<p>
The goal of the reward function as already said above is to favor a 
stable forward walk with maximum speed while staying stable and to penalize shaking, tumbling over etc.
For this the reward consists of several different terms which are:<br>
Foreward velocity reward at right pace<br>
Yaw reward for keeping the initial orientation<br>
Fixed reward for being upright which acts as a staying alive reward<br>
Two penalty terms if either the "belly" or "head" is touching something<br>
Penalty terms for velocity in another direction than foreward<br>
Penalty for rotational velocity of the base body<br>
Penalty for not having a stable and upright base body rotational position<br>
Penalty for having an unfavorable height of the base body<br>
Penalty for swaying from the base stance in the servo positions<br>
Penalty for servo velocities<br>
Penalty for actuator force<br>
Penalties for having changing actions for both velocity and acceleration<br>
<br>
Problems and possible solutions with each version:<br>
v0 - unstable shaky moving without clear direction and constant stumbling over
-> implement the yaw penalty, disabling the height penalty and adding the belly 
and head penalty instead, increasing rotational velocity penalty weight as well as rotational position weight
v1 - no real meassureable foreward velocity due to wrong speed goal updating -> added some plot logging and higher velocity
reward weight
v2 - robot has no idea how fast it is going or how fast it has to go -> added last action and speed goal to the observation
v3 - speed gain wasnt achieved rather stability with small movement was better -> changed the velocity reward and speed goal update rule
</p>