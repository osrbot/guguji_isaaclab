# Getting Started

## Requirements

Before installing this extension, make sure you have:

- Isaac Lab **2.1.0**
- Isaac Sim **4.5.0**
- Python **3.10**
- `rsl-rl-lib >= 5.0` available through the Isaac Lab environment

For Isaac Lab installation details, refer to the official guide:

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

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

## Verify environment registration

```bash
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/list_envs.py
```

You should see the four Guguji locomotion environments registered successfully.

## Preview the docs locally

The website uses MkDocs.

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

To produce a static build locally:

```bash
mkdocs build --strict
```

## Publish with GitHub Pages

A GitHub Actions workflow is included at `.github/workflows/docs.yml`.

To activate publishing in the repository settings:

1. Open **Settings** → **Pages**.
2. Set **Source** to **GitHub Actions**.
3. Push to `main` after editing `docs/`, `mkdocs.yml`, or the workflow file.

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
