# Getting Started

## Requirements

Before installing this extension, make sure you have:

- Isaac Lab **Latest**
- Isaac Sim **5.1.0**
- Python **3.10**
- `rsl-rl-lib >= 5.0` available through the Isaac Lab environment

For Isaac Lab installation details, refer to the official guide:

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

## Isaac Sim 5.1 + IsaacLab Environment Installation

!!! warning "Adjust commands to your local paths"
    The following commands use `~/rlgpu_ws/` as the example workspace:

    - Isaac Sim example path: `~/rlgpu_ws/IsaacSim`
    - IsaacLab example path: `~/rlgpu_ws/IsaacLab`

    If Isaac Sim or IsaacLab is already installed on your machine and the installation path is different, do not blindly copy and run every command. Before running each command that contains a path, replace the example paths with your actual local paths.

### Prebuilt package

Use the Isaac Sim 5.1 prebuilt package:

<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html>

### Installation steps

```bash
# 1. Create a workspace and download Isaac Sim
mkdir rlgpu_ws && cd rlgpu_ws
curl -LJO https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -o -d IsaacSim

# 2. Download IsaacLab
git clone https://github.com/isaac-sim/IsaacLab.git

# 3. Prepare IsaacLab environment variables
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

# 4. Install IsaacLab with the Python environment from the Isaac Sim prebuilt package
cd ~/rlgpu_ws/IsaacLab/
./isaaclab.sh -i              # Install the required IsaacLab base libraries
./isaaclab.sh --install all   # Install commonly used reinforcement learning libraries

# 5. Verify the IsaacLab environment
./isaaclab.sh -p scripts/environments/list_envs.py
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0
```

### Download local assets

Reference:

<https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#isaac-sim-setup-assets-content-pack>

The assets are split into multiple zip parts. Merge the parts first; directly unzipping the individual parts can fail.

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

### Configure the local asset path

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

### Test the configuration

```bash
cd ~/rlgpu_ws/IsaacSim
./isaac-sim.sh --/persistent/isaac/asset_root/default="/home/$USER/isaacsim_assets/Assets/Isaac/5.1"
```

## Clone the repository

```bash
cd ~/Desktop
git clone https://code.xturtle.cn/corvin_zhang/guguji_simulation.git
cd guguji_simulation/
git clone https://github.com/osrbot/guguji_isaaclab.git
```

## Install the extension into Isaac Lab

```bash
cd ~/rlgpu_ws/IsaacLab
./isaaclab.sh -p -m pip install -e ~/Desktop/guguji_simulation/guguji_isaaclab/source/guguji_locomotion
```

## Recommended workflow

1. Start with the flat-terrain task.
2. Wait for the policy to converge on straight, stable forward walking.
3. Move to rough terrain once the flat policy is reliable.
4. Use the `play.py` workflow to validate and export the resulting policy.

## Project layout

```text
source/guguji_locomotion/guguji_locomotion/
├── assets/
│   └── guguji.py
└── tasks/locomotion/velocity/
    ├── velocity_env_cfg.py
    ├── mdp/
    │   ├── actions.py
    │   ├── observations.py
    │   ├── rewards.py
    │   └── curriculums.py
    └── config/guguji/
        ├── __init__.py
        ├── flat_env_cfg.py
        ├── rough_env_cfg.py
        └── agents/rsl_rl_ppo_cfg.py
```

## Environment summary

| Environment | Role |
|---|---|
| `Isaac-Velocity-Flat-Guguji-v0` | Main training on flat terrain |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | Visualization and flat-terrain evaluation |
| `Isaac-Velocity-Rough-Guguji-v0` | Rough-terrain training |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | Rough-terrain evaluation |

## Notes

!!! note
    The repository is structured as an Isaac Lab extension. Install it with `pip install -e` inside the Isaac Lab Python environment rather than treating it as a standalone simulator project.

!!! tip
    Use the flat-terrain curriculum first. It is the intended entry point for faster debugging and more stable early-stage learning.
