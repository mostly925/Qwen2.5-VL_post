from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Task:
    task_id: str
    target_actions: str


class ToySparseControlEnv:
    """玩具稀疏奖励环境。

    设计目标：
    - 用最小代价模拟论文中的“稀疏反馈 + 延迟奖励”特点。
    - 保留文本反馈接口，供反思模块消费。
    """

    def __init__(self, action_space: str = "UDLR") -> None:
        self.action_space = action_space

    def evaluate(self, task: Task, attempt: str) -> Tuple[str, float]:
        target = task.target_actions
        if attempt == target:
            return "成功：所有步骤正确。", 1.0

        mismatch = self._first_mismatch(target, attempt)
        pos, expected, got = mismatch
        feedback = (
            f"失败：第{pos}步不正确；预期动作={expected}，实际动作={got}；"
            "请基于该错误进行修正。"
        )
        return feedback, 0.0

    @staticmethod
    def _first_mismatch(target: str, attempt: str) -> Tuple[int, str, str]:
        max_len = min(len(target), len(attempt))
        for idx in range(max_len):
            if target[idx] != attempt[idx]:
                return idx + 1, target[idx], attempt[idx]
        if len(attempt) < len(target):
            return len(attempt) + 1, target[len(attempt)], "<END>"
        return len(target) + 1, "<END>", attempt[len(target)]

    def valid_actions(self) -> List[str]:
        return list(self.action_space)

    def task_to_observation(self, task: Task) -> Dict[str, str]:
        return {"task_id": task.task_id, "length": str(len(task.target_actions))}
