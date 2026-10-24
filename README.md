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
<br>
v1 - no real meassureable foreward velocity due to wrong speed goal updating
-> added some plot logging and higher velocity<br>
reward weight<br>
<br>
v2 - robot has no idea how fast it is going or how fast it has to go<br>
-> added last action and speed goal to the observation<br>
<br>
v3 - relevant speed gain wasnt achieved rather stability increased with small movement<br>
-> changed the velocity reward and speed goal update rule<br>
<br>
v4 - still issues with velocity not increasing <br>
-> changing vel reward to scale with velocity instead of goal speed and slighty increasing
rotational penalty weights<br>
<br>
v5 - unstable walking resulting in abrupt small stops between steps and occasional falling over<br>
-> increasing rotational velocity penalty weight and rotational position penalty weight<br>
<br>
v6 - walking in circles and still small unstabilities<br>
-> connect the penalty weights for stability with foreward velocity to emphasize stability when walking fast
<br>
<br>
v7 - still walking in circles but more stables at around 0.7m/s<br>
- Changes:<br>
-> The velocity tracked now is not the velocity from the body frame but the world frame rotated to the desired yaw direction<br>
-> A small starting help was added in form of the velocity in the bodies direction<br>
-> A large part of the reward consists of if the base is oriented in the right direction and not just running into the direction<br>
-> The error from the yaw to the goal orientation was added into the observation<br>
-> The reset of the mujoco was fixed as the older version kept information from previous episodes like rotation<br>
-> The logging has been expanded to all almost all penalty and reward terms and changed to a pandas dataframe<br>
-> A new termination possibility by touching the floor with the belly has been added as test runs had the problem that the robot would catch the front edge of the base and stumble or fall over after initial progress<br>
- Results:<br>
-> The Robot closely now tracks the desired direction with a speed up to 1.2 m/s with rarely falling over<br>
-> Sometimes it somewhat stumbles a bit and has to catch itself first which results in a bit of offset in yaw and temporare increase in rotational velocity<br>
-> Better insights from the plots by using the episode average instead of the data from each step<br>
<br>
v8 - still a little bit unsteady and recoveries aren't "instant"<br>
- Changes:<br>
-> increased the number of training steps from 2.5 mil to 5 mil and added learning rate decay as well as callbacks in case of degradation<br>
- Results:<br>
-> speed up to 1.45 m/s with high kicking front legs and a bit faster recoveries<br>
<br>
v9 - actuator data is not matching ones that could be used for hobby projects<br>
-> change the specifications of the actuators to match Waveshare ST3215-HS servos as the current gait utilizes the maximum capacities of the actuators<br>
- Occuring problems:<br>
-> due to the lower actuator force the robot used one knee on the ground or small "falls" to stablelize itself<br>
-> the knee problem could be fixed by adding a knee touch penalty<br>
-> the resulting stuttering or falling was improved by adding a acceleration penalty<br>
- Other changes include:<br>
-> adding knee sensors to the robot<br>
-> changing the joint range and decreased the force of the actuators<br>
- Results:<br>
-> v9 can walk stable now with a smooth continious gait<br>
-> the bot can now walk up to around 1 m/s with closely tracking a set yaw<br>
<br>
v10 - the analysis script shows that the robot demands higher torque most of the time and the high swings of the front legs results in slams into the ground<br>
- Changes:<br>
-> adding sensors on the feet and adding penalty terms for the slams and stiffening the floor<br>
-> adding a penalty for demanding more torque than can be delivered<br>
-> changed the env to track max vertical speed of the feet and divided the sim steps into substeps<br>
-> changed the randomization ramp changed from episodes to steps and earlier max out<br>
- occuring problems:<br>
-> the analysis showed that the feet are also sliding a bit while having contact with the ground<br>
-> the slams occur fast due to the high velocities therefore the simulation steps had to be split up without slowing down training<br>
- Results:<br>
-> the swing height of the front legs could be decreased to a certain degree<br>
-> the sliding of the feet while standing could be decreased resulting in overall cleaner gait in the plots<br>
-> the torque problem couldnt be fixed properly as the resulting speed reward is greater than the penalty from a higher torque<br>
<br>
v11 - the reward term has to be overworked to enforce the torque limits on the policies outputs<br>
- Changes:<br>
-> Changed the kp and kv values for the actuators for better force control when handling noise<br>
-> Changed the control range for a possible better gait while staying in a possible range<br>
-> Base reward weight decays with the down to 1 for better initial learning and less "distraction" in later training<br>
-> Yaw reward increase with ramp factor for better start<br>
-> Increased the touch penalt weights to stop "self-harming" behaviour direclty when occuring<br>
-> Scaled the action space back down to -2 to 2 for more tolerance while learning<br>
-> Changed the force_ac penalty so that the used force isnt 100% of the maximum available force<br>
-> Slight increase in the weight of slam and sliding penalty<br>
- occuring problems:<br>
-> a faulty force penalty can result in a policy that cant really move as a high force usage penalizes higher than a stable speed gain<br>
-> less stiff servos indireclty led to a policy learning to use the one knee instead of the foot<br>
-> too high reward terms lead to a policy trying everything to gain a bit more speed instead of not overdemanding the servos<br>
-> too high total penalty resulted in the reward drifting below 0 which resulted in the policy not effectively learning anymore<br>
- Results:<br>
-> the policy learned fast how to run quite early<br>
-> later progress slowed as initial high actions and higher servo force resulted in higher penalties<br>
-> in the end it reached around 1.0 m/s and problems like torque and action penalty decreased<br>
-> analysis show higher front foot swing but a slower vertical descent as well as less servo demand<br>
-> gait has been analyzed for v10 under the new and old robot config for at least some comparability<br>

