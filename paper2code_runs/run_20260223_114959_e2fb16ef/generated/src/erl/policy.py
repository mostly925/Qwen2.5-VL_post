import math
import random
from typing import Dict, List, Tuple

from .environment import Task


class TabularPolicy:
    """按 task_id 与 step 维护动作分布的表格策略。"""

    def __init__(self, actions: List[str], seed: int = 7) -> None:
        self.actions = actions
        self.rng = random.Random(seed)
        self.logits: Dict[Tuple[str, int], Dict[str, float]] = {}

    def sample_attempt(self, task: Task) -> str:
        result = []
        for step_idx in range(len(task.target_actions)):
            action = self._sample_action(task.task_id, step_idx)
            result.append(action)
        return "".join(result)

    def reflection(
        self, task: Task, first_attempt: str, feedback: str, memory_hint: str
    ) -> str:
        """基于反馈生成结构化反思文本。"""
        _ = task, first_attempt, memory_hint
        if "第" not in feedback or "预期动作=" not in feedback:
            return "未解析到错误位置信息，请复查整体策略。"

        pos = feedback.split("第", 1)[1].split("步", 1)[0]
        expected = feedback.split("预期动作=", 1)[1].split("，", 1)[0]
        return f"反思：第{pos}步动作应改为{expected}，其余步骤保持不变。"

    def refined_attempt(self, task: Task, first_attempt: str, reflection: str) -> str:
        """根据反思修正 first attempt，得到 second attempt。"""
        if "第" not in reflection or "步动作应改为" not in reflection:
            return first_attempt

        pos_text = reflection.split("第", 1)[1].split("步", 1)[0]
        expected = reflection.split("步动作应改为", 1)[1].split("，", 1)[0]
        try:
            pos = int(pos_text)
        except ValueError:
            return first_attempt

        idx = pos - 1
        if idx < 0 or idx >= len(first_attempt):
            return first_attempt

        chars = list(first_attempt)
        chars[idx] = expected
        return "".join(chars)

    def reinforce_update(
        self, task: Task, attempt: str, reward: float, lr: float
    ) -> None:
        """近似策略梯度更新：奖励高则提高采样动作 logit，反之降低。"""
        advantage = reward - 0.5
        for step_idx, action in enumerate(attempt):
            bucket = self._get_bucket(task.task_id, step_idx)
            bucket[action] += lr * advantage

    def distill_update(
        self, task: Task, target_attempt: str, distill_rate: float
    ) -> None:
        """内化蒸馏：将成功 second attempt 压入 base policy。"""
        for step_idx, action in enumerate(target_attempt):
            bucket = self._get_bucket(task.task_id, step_idx)
            bucket[action] += distill_rate

    def greedy_attempt(self, task: Task) -> str:
        result = []
        for step_idx in range(len(task.target_actions)):
            bucket = self._get_bucket(task.task_id, step_idx)
            best = max(bucket.items(), key=lambda kv: kv[1])[0]
            result.append(best)
        return "".join(result)

    def _sample_action(self, task_id: str, step_idx: int) -> str:
        bucket = self._get_bucket(task_id, step_idx)
        probs = self._softmax(bucket)
        pivot = self.rng.random()
        cumulative = 0.0
        for action in self.actions:
            cumulative += probs[action]
            if pivot <= cumulative:
                return action
        return self.actions[-1]

    def _get_bucket(self, task_id: str, step_idx: int) -> Dict[str, float]:
        key = (task_id, step_idx)
        if key not in self.logits:
            self.logits[key] = {a: 0.0 for a in self.actions}
        return self.logits[key]

    @staticmethod
    def _softmax(bucket: Dict[str, float]) -> Dict[str, float]:
        max_logit = max(bucket.values())
        exp_vals = {k: math.exp(v - max_logit) for k, v in bucket.items()}
        total = sum(exp_vals.values())
        return {k: v / total for k, v in exp_vals.items()}
