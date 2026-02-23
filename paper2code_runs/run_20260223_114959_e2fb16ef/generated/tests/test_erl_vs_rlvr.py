import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.erl import (
    ERLConfig,
    ERLTrainer,
    RLVRTrainer,
    TabularPolicy,
    Task,
    ToySparseControlEnv,
    evaluate_policy,
)


class TestERLVersusRLVR(unittest.TestCase):
    def test_erl_reward_not_worse_than_rlvr(self) -> None:
        tasks = [
            Task(task_id="t1", target_actions="UURD"),
            Task(task_id="t2", target_actions="LLDR"),
            Task(task_id="t3", target_actions="RDLU"),
            Task(task_id="t4", target_actions="DRUL"),
        ]
        env = ToySparseControlEnv(action_space="UDLR")
        cfg = ERLConfig(
            episodes=500, tau=0.8, learning_rate=0.12, distill_rate=0.35, seed=42
        )

        erl_policy = TabularPolicy(actions=env.valid_actions(), seed=42)
        erl = ERLTrainer(env=env, policy=erl_policy, config=cfg)
        erl.train(tasks)
        erl_score = evaluate_policy(env, erl_policy, tasks)

        rlvr_policy = TabularPolicy(actions=env.valid_actions(), seed=42)
        rlvr = RLVRTrainer(env=env, policy=rlvr_policy, config=cfg)
        rlvr.train(tasks)
        rlvr_score = evaluate_policy(env, rlvr_policy, tasks)

        self.assertGreaterEqual(erl_score, rlvr_score)


if __name__ == "__main__":
    unittest.main()
