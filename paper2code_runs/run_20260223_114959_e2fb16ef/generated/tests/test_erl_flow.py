import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.erl import ERLConfig, ERLTrainer, TabularPolicy, Task, ToySparseControlEnv


class TestERLFlow(unittest.TestCase):
    def test_memory_is_written_when_second_reward_exceeds_tau(self) -> None:
        tasks = [Task(task_id="a", target_actions="UU")]
        env = ToySparseControlEnv(action_space="UDLR")
        cfg = ERLConfig(
            episodes=300, tau=0.8, learning_rate=0.2, distill_rate=0.4, seed=1
        )
        policy = TabularPolicy(actions=env.valid_actions(), seed=1)

        trainer = ERLTrainer(env=env, policy=policy, config=cfg)
        trainer.train(tasks)

        self.assertGreater(len(trainer.memory), 0)


if __name__ == "__main__":
    unittest.main()
