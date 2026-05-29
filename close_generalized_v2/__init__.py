#!/usr/bin/env python3
# close_generalized_v2/__init__.py
#
# Advanced Generalization of the Door Closing Task.
#
# Modules:
#   config_v2        — extended TrainConfigV2 dataclass
#   fsm_v2           — AdaptiveFSM: context-sensitive transition thresholds  [§3.1]
#   reward_v2        — PotentialBasedReward: hierarchical potential shaping  [§3.2]
#   grasp_strategy   — MultiApproachGrasp: K-candidate alignment             [§3.3]
#   domain_rand_v2   — ExtendedDomainRandomizer: physics randomization       [§3.4]
#   beta_net         — BetaNetwork: learned termination functions            [§3.5]
#   env_v2           — AdvancedGeneralizedDoorEnv: top-level environment     (all)
#   train_gen_v2     — Training entry-point                                  (all)

from close_generalized_v2.env_v2 import AdvancedGeneralizedDoorEnv

__all__ = ["AdvancedGeneralizedDoorEnv"]