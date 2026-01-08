Visual positioning using 3DGS+VINS

## 1. Prerequisites
### 1.1 **Ubuntu**, **ROS**, **CUDA**
Tested on Ubuntu 20.04 with ROS Neotic and CUDA 11.8.
### 1.2. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).
### 1.3. **requirements**
pip install -r environment.yml
Basically, all you need is a CUDA environment that can run VINS-Fusion and a 3D Gaussian Splatting environment.
You can refer to the VINS and 3DGS
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
## 2. Build VINS-GS & train your 3DGS model
Clone the repository and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/JinXJinLeX/VINS-GS.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```
Train your 3DGS model:
```
    conda activate VINS-GS 
    cd src/splatting
    python3 train.py -s YOUR_DATABSE_PATH -m YOUR_MODEL_PATH
```
## 3. Run VINS-GS
### 3.1 Set path
```
    cd src/splatting/arguments 
    gedit __init__.py
```
change the path to your model path
```
    def get_combined_args(parser : ArgumentParser):
        cmdlne_string = sys.argv[1:]
        cmdlne_string[0]="-m"
        #this is the path to your model
        cmdlne_string[1]="/home/seu/xjl_work_space/gaussian-splatting/data/kasitVIO/square/square_head_infra1/output" 
```
change the path to your database path
```
    cd src/VINS-Fusion/gaussian_map/scripts
    gedit gaussian_map.py
```
there is a parameter N_ITERATION, change it to 30k
N_ITERATION = 90000
yaml_path = "/home/seu/xjl_work_space/VINS-Fusion/src/VINS-Fusion/config/3DGS_kasit/camera.yaml"
change the path to your camera.yaml
### 3.2 Set yaml
We change the config for 3DGS, make sure you have them:
```
    gaussian_map: 1 #1: use GS #2 not
    sp_sg: 0 #1: use SuperpPoint #0: not
    orb: 1 #1: use ORB #0: not
    map_yaw: 0 # initial orientation
    map_pitch: 0
    map_roll: 0
    map_initial_position_x: -0.131 #initial position
    map_initial_position_y: -0.013
    map_initial_position_z: 0.004
    groundtruth_file: "/home/seu/xjl_work_space/VINS-Fusion/output/kasitVIO/square/square_head/groundtruth.txt" # path of groundtruth.txt for visualization
```
### 3.3 Run
You can run VINS-GS by:
```
    cd ~/catkin_ws
    chmod +x launch_vins_gaussian.sh
    ./launch_vins_gaussian.sh
```

Or open a terminal and run:
```
    cd ~/catkin_ws
    conda activate VINS-GS
    source devel/setup.bash
    roslaunch gaussian_map gaussian_map.launch
```
Then open a new terminal and run:
```
    cd ~/catkin_ws
    conda activate VINS-GS
    source devel/setup.bash
    rosrun vins vins_node src/VINS-Fusion/config/3DGS_kasit/square_head.yaml 
```
For now, the superpoint and superglue are not stable, so set orb to 1.