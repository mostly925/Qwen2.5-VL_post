import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.erl import (
    ERLConfig,
    ERLTrainer,
    RLVRTrainer,
    TabularPolicy,
    Task,
    ToySparseControlEnv,
    evaluate_policy,
)


def build_tasks() -> list[Task]:
    return [
        Task(task_id="t1", target_actions="UURD"),
        Task(task_id="t2", target_actions="LLDR"),
        Task(task_id="t3", target_actions="RDLU"),
        Task(task_id="t4", target_actions="DRUL"),
    ]


def main() -> None:
    tasks = build_tasks()
    env = ToySparseControlEnv(action_space="UDLR")

    config = ERLConfig(
        episodes=600, tau=0.8, learning_rate=0.12, distill_rate=0.35, seed=42
    )

    erl_policy = TabularPolicy(actions=env.valid_actions(), seed=42)
    erl_trainer = ERLTrainer(env=env, policy=erl_policy, config=config)
    erl_stats = erl_trainer.train(tasks)
    erl_eval = evaluate_policy(env, erl_policy, tasks)

    rlvr_policy = TabularPolicy(actions=env.valid_actions(), seed=42)
    rlvr_trainer = RLVRTrainer(env=env, policy=rlvr_policy, config=config)
    rlvr_stats = rlvr_trainer.train(tasks)
    rlvr_eval = evaluate_policy(env, rlvr_policy, tasks)

    print("=== ERL 最小复现结果 ===")
    print(
        f"ERL 训练均值(first/second): {erl_stats.first_reward_mean:.3f}/{erl_stats.second_reward_mean:.3f}"
    )
    print(f"RLVR 训练均值(first): {rlvr_stats.first_reward_mean:.3f}")
    print(f"ERL 贪心评估奖励: {erl_eval:.3f}")
    print(f"RLVR 贪心评估奖励: {rlvr_eval:.3f}")
    print(f"ERL 记忆条目数: {len(erl_trainer.memory)}")


if __name__ == "__main__":
    main()
