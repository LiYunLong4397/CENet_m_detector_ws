#!/bin/bash

# 设置环境
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# 下载CENet预训练模型（如果还没有）
if [ ! -f "/path/to/CENet/checkpoints/model_best.pth" ]; then
    echo "请先下载CENet预训练模型"
    exit 1
fi

# 启动roscore
roscore &
sleep 2

# 启动M-detector with semantic
roslaunch m_detector dynfilter_semantic.launch &

# 播放测试bag
rosbag play /path/to/test. bag

# 等待完成
wait