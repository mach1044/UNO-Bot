from abc import ABC, abstractmethod


class Bot(ABC):
    @abstractmethod
    def choose_move(
        self,
        observation: dict,
        legal_moves: list[list[object]],
    ) -> tuple[list[object], str | None]:
        pass
