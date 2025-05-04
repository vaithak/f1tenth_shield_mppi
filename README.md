## MPPI with Control Barrier Functions (Shield MPPI) for F1Tenth car

### Building: 
1. Clone this repo
2. This repo uses some custom message types for communication, defined here: https://github.com/vaithak/f1tenth_icra_race_msgs
   ```
   git clone git@github.com:vaithak/f1tenth_icra_race_msgs.git
   colcon build --packages-select=f1tenth_icra_race_msgs
   ```
3. Build the package
   ```
   colcon build
   source install/local_setup.bash
   ```

### Running
- Launch the object and tracking subsystems
  `ros2 launch f1tenth_icra_race r3.oo.launch.py`
- Launch the spliner and state machine
  `ros2 launch f1tenth_icra_race r3.ss.launch.py`
- Launch the shield mppi controller
  `ros2 launch f1tenth_icra_race r3.c.launch.py`

### Subsystems and workings


### Working demo on sim
[shield mppi](https://github.com/user-attachments/assets/b5d3f740-e091-4092-b12f-3779751e319c)

