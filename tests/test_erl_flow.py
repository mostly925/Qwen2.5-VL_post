import unittest

from trainer.erl import ERLConfig, ERLTrainer, TabularPolicy, Task, ToySparseControlEnv


class TestERLFlow(unittest.TestCase):
    def test_memory_write_when_second_reward_above_tau(self) -> None:
        env = ToySparseControlEnv(action_space="UDLR")
        policy = TabularPolicy(actions=env.valid_actions(), seed=42)
        config = ERLConfig(
            episodes=200, tau=0.8, learning_rate=0.12, distill_rate=0.35, seed=42
        )
        trainer = ERLTrainer(env=env, policy=policy, config=config)
        tasks = [Task(task_id="t1", target_actions="UURD")]

        trainer.train(tasks)

        self.assertGreater(len(trainer.memory), 0)


if __name__ == "__main__":
    unittest.main()
