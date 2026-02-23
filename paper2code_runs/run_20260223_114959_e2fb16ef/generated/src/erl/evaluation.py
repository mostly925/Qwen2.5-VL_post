from typing import List

from .environment import Task, ToySparseControlEnv
from .policy import TabularPolicy


def evaluate_policy(
    env: ToySparseControlEnv, policy: TabularPolicy, tasks: List[Task]
) -> float:
    rewards = []
    for task in tasks:
        attempt = policy.greedy_attempt(task)
        _, reward = env.evaluate(task, attempt)
        rewards.append(reward)
    return sum(rewards) / len(rewards)
