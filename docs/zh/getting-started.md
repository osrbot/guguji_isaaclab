# 中文快速开始

## 环境要求

在安装本扩展前，请确保你已经具备：

- Isaac Lab **Latest**
- Isaac Sim **5.1.0**
- Python **3.10**
- 在 Isaac Lab 环境中可用的 `rsl-rl-lib >= 5.0`

Isaac Lab 安装说明请参考官方文档：

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

## Isaac Sim 5.1 + IsaacLab 环境安装

!!! warning "请根据本机路径调整命令"
    本文后续命令以 `~/rlgpu_ws/` 作为示例工作目录：

    - Isaac Sim 示例路径：`~/rlgpu_ws/IsaacSim`
    - IsaacLab 示例路径：`~/rlgpu_ws/IsaacLab`

    如果你已经安装过 Isaac Sim 或 IsaacLab，且安装路径不同，请不要直接复制执行所有命令。运行每一行涉及路径的命令前，请先将示例路径替换为你本机的实际路径。

### 预编译安装包

使用 Isaac Sim 5.1 预编译安装包：

<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html>

### 安装过程

```bash
# 1. 创建 workspace 和下载 Isaac Sim
mkdir rlgpu_ws && cd rlgpu_ws
curl -LJO https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -o -d IsaacSim

# 2. 下载 IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git

# 3. 做好 IsaacLab 的环境变量准备工作
cd IsaacLab && ln -s ../IsaacSim/ _isaac_sim

cat <<'EOF' >> ~/.bashrc

isaacsim () {
    # Isaac Sim root directory
    export ISAACSIM_PATH="${HOME}/rlgpu_ws/IsaacSim"
    # Isaac Sim python executable
    export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
    export PATH=$PATH:${ISAACSIM_PATH}/kit/python/bin
    source ${ISAACSIM_PATH}/setup_conda_env.sh
}
EOF

source ~/.bashrc
isaacsim

# 4. 基于 Isaac Sim 预编译的 Python 环境安装 IsaacLab
cd ~/rlgpu_ws/IsaacLab/
./isaaclab.sh -i              # 安装必要的 IsaacLab 基础库
./isaaclab.sh --install all   # 安装常用的强化学习库

# 5. 验证 IsaacLab 环境
./isaaclab.sh -p scripts/environments/list_envs.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0
```

### 本地资产下载安装

参考文档：

<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>

注意 zip 文件需要先合并，直接分别解压分卷 zip 可能会报错。

```bash
sudo apt install aria2
cd ~/Downloads
aria2c "https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.001.zip"
aria2c "https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.002.zip"
aria2c "https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-5.1.0.003.zip"

mkdir ~/isaacsim_assets
cd ~/Downloads
cat isaac-sim-assets-complete-5.1.0.001.zip isaac-sim-assets-complete-5.1.0.002.zip isaac-sim-assets-complete-5.1.0.003.zip > isaac-sim-assets-complete-5.1.0.zip
unzip "isaac-sim-assets-complete-5.1.0.zip" -d ~/isaacsim_assets
```

### 设置本地资产路径

```bash
cat <<EOF >> ~/rlgpu_ws/IsaacSim/apps/isaacsim.exp.base.kit
[settings]
persistent.isaac.asset_root.default = "/home/$USER/isaacsim_assets/Assets/Isaac/5.1"

exts."isaacsim.gui.content_browser".folders = [
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/People",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Materials",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Samples",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Sensors",
]

# Optional: Using Content Browser is recommended.
exts."isaacsim.asset.browser".folders = [
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/People",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/IsaacLab",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Props",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Materials",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Samples",
    "/home/$USER/isaacsim_assets/Assets/Isaac/5.1/Isaac/Sensors",
]
EOF

grep "/home/" ~/rlgpu_ws/IsaacSim/apps/isaacsim.exp.base.kit
```

### 测试配置正确

```bash
cd ~/rlgpu_ws/IsaacSim
./isaac-sim.sh --/persistent/isaac/asset_root/default="/home/$USER/isaacsim_assets/Assets/Isaac/5.1"
```

## 克隆仓库

```bash
cd ~/Desktop
git clone https://code.xturtle.cn/corvin_zhang/guguji_simulation.git
cd guguji_simulation/
git clone https://github.com/osrbot/guguji_isaaclab.git
```

## 将扩展安装到 Isaac Lab

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m pip install -e ~/Desktop/guguji_simulation/guguji_isaaclab/source/guguji_locomotion
```

## 推荐使用顺序

1. 先从平地任务开始
2. 确认机器人已经能稳定直行
3. 再切到粗糙地形训练
4. 用 `play.py` 做可视化验证与策略导出
