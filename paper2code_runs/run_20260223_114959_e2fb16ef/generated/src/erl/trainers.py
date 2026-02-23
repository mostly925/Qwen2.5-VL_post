import random
from dataclasses import dataclass
from typing import List

from .config import ERLConfig
from .environment import Task, ToySparseControlEnv
from .memory import MemoryItem, ReflectionMemory
from .policy import TabularPolicy


@dataclass
class TrainStats:
    first_reward_mean: float
    second_reward_mean: float


class RLVRTrainer:
    """标准 RLVR 基线：单次尝试 + 奖励驱动更新。"""

    def __init__(
        self, env: ToySparseControlEnv, policy: TabularPolicy, config: ERLConfig
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = config
        self.rng = random.Random(config.seed)

    def train(self, tasks: List[Task]) -> TrainStats:
        first_rewards: List[float] = []
        for _ in range(self.config.episodes):
            task = self.rng.choice(tasks)
            attempt = self.policy.sample_attempt(task)
            _, reward = self.env.evaluate(task, attempt)
            first_rewards.append(reward)
            self.policy.reinforce_update(
                task, attempt, reward, self.config.learning_rate
            )

        mean_first = sum(first_rewards) / len(first_rewards)
        return TrainStats(first_reward_mean=mean_first, second_reward_mean=mean_first)


class ERLTrainer:
    """ERL 训练器：双尝试、反思、记忆与内化蒸馏。"""

    def __init__(
        self, env: ToySparseControlEnv, policy: TabularPolicy, config: ERLConfig
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = config
        self.memory = ReflectionMemory(max_size=config.memory_max_size)
        self.rng = random.Random(config.seed)

    def train(self, tasks: List[Task]) -> TrainStats:
        first_rewards: List[float] = []
        second_rewards: List[float] = []

        for _ in range(self.config.episodes):
            task = self.rng.choice(tasks)

            first_attempt = self.policy.sample_attempt(task)
            first_feedback, first_reward = self.env.evaluate(task, first_attempt)
            first_rewards.append(first_reward)

            if first_reward < self.config.tau:
                memory_hint = self._memory_hint(task.task_id)
                reflection = self.policy.reflection(
                    task, first_attempt, first_feedback, memory_hint
                )
                second_attempt = self.policy.refined_attempt(
                    task, first_attempt, reflection
                )
            else:
                reflection = "首轮已达阈值，跳过反思。"
                second_attempt = first_attempt

            _, second_reward = self.env.evaluate(task, second_attempt)
            second_rewards.append(second_reward)

            reflection_reward = second_reward

            if second_reward > self.config.tau:
                self.memory.add(
                    MemoryItem(
                        task_id=task.task_id,
                        reflection=reflection,
                        reward_second=second_reward,
                    )
                )

            self.policy.reinforce_update(
                task, first_attempt, first_reward, self.config.learning_rate
            )
            self.policy.reinforce_update(
                task, second_attempt, reflection_reward, self.config.learning_rate
            )

            if second_reward > 0.0:
                self.policy.distill_update(
                    task, second_attempt, self.config.distill_rate
                )

        return TrainStats(
            first_reward_mean=sum(first_rewards) / len(first_rewards),
            second_reward_mean=sum(second_rewards) / len(second_rewards),
        )

    def _memory_hint(self, task_id: str) -> str:
        items = [m for m in self.memory.latest(limit=5) if m.task_id == task_id]
        if not items:
            return ""
        return "\n".join(x.reflection for x in items)
