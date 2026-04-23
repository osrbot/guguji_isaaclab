# 中文快速开始

## 环境要求

在安装本扩展前，请确保你已经具备：

- Isaac Lab **2.1.0**
- Isaac Sim **4.5.0**
- Python **3.10**
- 在 Isaac Lab 环境中可用的 `rsl-rl-lib >= 5.0`

Isaac Lab 安装说明：

- <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>

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

## 验证环境注册

```bash
./isaaclab.sh -p ~/Desktop/guguji_simulation/guguji_isaaclab/scripts/list_envs.py
```

如果安装正确，你应该能看到四个 Guguji 相关环境已经注册。

## 本地预览网站

文档网站基于 MkDocs。

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

然后在浏览器中打开 `http://127.0.0.1:8000`。

本地静态构建：

```bash
mkdocs build --strict
```

## 发布到 GitHub Pages

仓库已经包含 `.github/workflows/docs.yml` 自动部署工作流。

在仓库设置中还需要手动开启：

1. 打开 **Settings** → **Pages**
2. 将 **Source** 设置为 **GitHub Actions**
3. 后续只要修改 `docs/`、`mkdocs.yml` 或 workflow 文件并推送到 `main`，网站就会自动发布

## 推荐使用顺序

1. 先从平地任务开始
2. 确认机器人已经能稳定直行
3. 再切到粗糙地形训练
4. 用 `play.py` 做可视化验证与策略导出
