<h1>quadruped-rl</h1>
<h2>Goal</h2>
<p>
The goal of this project is to build a quadruped robot for a simulated environment and training it to walk by using reinforcement learning.
While doing the project i will log my progress and various training metrics for an analysis of the training and the end-result.
</p>
<h2>The Robot</h2>
<p>
The Robot is a 12 DOF quadruped robot which is structured similiar to spiders. For the physics engine MuJoCo will be used due to it being used for similar projects 
and its easier handling compared to other ones. To simulate the robot we first need a digital version of it as Mujoco works with mjcf files which are written in xml.
</p>
<h2>The robots specifications</h2>
<p>
The base is: 16cm x 16cm x 4cm<br>
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
<p>
v0 - unstable shaky moving without clear direction and constant stumbling over<br>
-> implement the yaw penalty, disabling the height penalty and adding the belly 
and head penalty instead, increasing rotational velocity penalty weight as well as rotational position weight<br>
v1 - no real meassureable foreward velocity due to wrong speed goal updating
-> added some plot logging and higher velocity<br>
reward weight<br>
v2 - robot has no idea how fast it is going or how fast it has to go<br>
-> added last action and speed goal to the observation<br>
v3 - relevant speed gain wasnt achieved rather stability increased with small movement<br>
-> changed the velocity reward and speed goal update rule<br>
v4 - still issues with velocity not increasing <br>
-> changing vel reward to scale with velocity instead of goal speed and slighty increasing
rotational penalty weights<br>
v5 - unstable walking resulting in abrupt small stops between steps and occasional falling over<br>
-> increasing rotational velocity penalty weight and rotational position penalty weight<br>
v6 - walking in circles and still small unstabilities<br>
-> connect the penalty weights for stability with foreward velocity to emphasize stability when walking fast
<br>
v7 - still walking in circles but more stables at around 0.7m/s<br>
&nbsp;&nbsp;&nbsp;- Code Changes to v7:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity tracked now is not the velocity from the body frame but the world frame rotated to the desired yaw direction<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A small starting help was added in form of the velocity in the bodies direction<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A large part of the reward consists of if the base is oriented in the right direction and not just running into the direction<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The error from the yaw to the goal orientation was added into the observation<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The reset of the mujoco was fixed as the older version kept information from previous episodes like rotation<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The logging has been expanded to all almost all penalty and reward terms and changed to a pandas dataframe<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A new termination possibility by touching the floor with the belly has been added as test runs had the problem that the robot would catch the front edge of the base and stumble or fall over after initial progress<br>
&nbsp;&nbsp;&nbsp;- Results from the changes:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The Robot closely now tracks the desired direction with a speed up to 1.2 m/s with rarely falling over<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sometimes it somewhat stumbles a bit and has to catch itself first which results in a bit of offset in yaw and temporare increase in rotational velocity<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Better insights from the plots by using the episode average instead of the data from each step<br>
v8 - still a little bit unsteady and recoveries aren't "instant"<br>
-> increased the number of training steps from 2.5 mil to 5 mil and added learning rate decay as well as callbacks in case of degradation<br>
-> Results: speed up to 1.45 m/s with high kicking front legs and a bit faster recoveries<br>
v9 - actuator data is not matching ones that could be used for hobby projects irl<br>
-> change the specifications of the actuators to match Waveshare ST3215-HS servos as the current gait utilizes the maximum capacities of the actuators<br>
