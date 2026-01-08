#!/bin/bash

# 核心：一条命令打开单窗口，同时创建两个标签页
gnome-terminal \
  --tab -t "Gaussian_Map" -e "bash -c 'source ~/.bashrc; source ~/anaconda3/etc/profile.d/conda.sh; cd ~/xjl_work_space/VINS-Fusion; source devel/setup.bash; conda activate VINS-Fusion; roslaunch gaussian_map gaussian_map.launch; exec bash'" \
  --tab -t "VINS_Node"   -e "bash -c 'source ~/.bashrc; source ~/anaconda3/etc/profile.d/conda.sh; cd ~/xjl_work_space/VINS-Fusion; source devel/setup.bash; conda activate VINS-Fusion; rosrun vins vins_node src/VINS-Fusion/config/3DGS_kasit/square_head.yaml; exec bash'"

echo "✅ GS-VINS已启动！"
