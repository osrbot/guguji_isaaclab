# Tested Environments

This page documents the hardware and software configurations that have been validated with this project.

## Hardware

| Component | Specification |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) x86_64 |
| Kernel | Linux 6.17.0-22-generic |
| CPU | AMD Ryzen 9 7950X3D (32 threads) @ 5.76 GHz |
| GPU (training) | NVIDIA GeForce RTX 4080 SUPER |
| GPU (integrated) | AMD Raphael |
| RAM | 32 GB DDR5 |

## Software

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | Latest (main) |
| RSL-RL | >= 5.0 |
| Python | 3.10 |
| ROS 2 | Jazzy |
| CUDA | 12.x (bundled with Isaac Sim) |

## Simulation Environments

All four registered Gym environments have been validated on the above stack:

| Environment ID | Terrain | Purpose |
|---|---|---|
| `Isaac-Velocity-Flat-Guguji-v0` | Flat | Training |
| `Isaac-Velocity-Flat-Guguji-Play-v0` | Flat | Evaluation / visualization |
| `Isaac-Velocity-Rough-Guguji-v0` | Rough | Training |
| `Isaac-Velocity-Rough-Guguji-Play-v0` | Rough | Evaluation / visualization |

## Notes

!!! note
    Isaac Sim 5.1.0 ships its own Python 3.10 interpreter and CUDA runtime. Do not mix system Python with the Isaac Lab environment.

!!! tip
    Training on the RTX 4080 SUPER with 2048 parallel environments runs at roughly 10–15k FPS on flat terrain. Rough terrain is slightly slower due to height-scan computation.
