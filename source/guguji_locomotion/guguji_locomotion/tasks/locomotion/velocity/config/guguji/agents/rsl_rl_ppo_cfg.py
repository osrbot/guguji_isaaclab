"""RSL-RL PPO agent configs for Guguji biped locomotion.

Uses the rsl_rl >= 4.0 API: separate actor/critic RslRlMLPModelCfg.
handle_deprecated_rsl_rl_cfg in train.py migrates stochastic/init_noise_std
to distribution_cfg automatically for rsl_rl >= 5.0.

Tuned to match the SB3 walk_ppo.yaml training parameters:
  - 3-stage curriculum: 0.18 -> 0.22 -> 0.26 m/s
  - Smaller network than AnymalD (biped is simpler, fewer joints)
  - Tighter clip range (0.15) and lower entropy coef (0.0)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class GugujiFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "guguji_flat"
    empirical_normalization = False

    # Empty dict: resolve_obs_groups auto-maps actor/critic to the "policy" group
    obs_groups = {}

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 256, 128],
        activation="elu",
        stochastic=True,
        init_noise_std=1.0,
    )
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[256, 256, 128],
        activation="elu",
        stochastic=False,
        init_noise_std=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=6.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class GugujiRoughPPORunnerCfg(GugujiFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "guguji_rough"
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]

