from collections import deque
from dataclasses import dataclass
from typing import Deque, List


@dataclass
class MemoryItem:
    task_id: str
    reflection: str
    reward_second: float


class ReflectionMemory:
    """跨回合反思记忆池。

    该类对应论文中的跨 episode memory，负责保存高质量反思，
    并为后续反思提供先验上下文。
    """

    def __init__(self, max_size: int) -> None:
        self._items: Deque[MemoryItem] = deque(maxlen=max_size)

    def add(self, item: MemoryItem) -> None:
        self._items.append(item)

    def latest(self, limit: int = 5) -> List[MemoryItem]:
        return list(self._items)[-limit:]

    def __len__(self) -> int:
        return len(self._items)
