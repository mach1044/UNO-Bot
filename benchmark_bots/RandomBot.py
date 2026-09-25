import random

from engine.Bot import Bot
from engine.Card import Card


class RandomBot(Bot):
    def choose_move(
        self,
        observation: dict,
        legal_moves: list[list[object]],
    ) -> tuple[list[object], str | None]:
        if not legal_moves:
            raise ValueError("The bot was given no legal moves")

        move = random.choice(legal_moves)
        chosen_color = None

        if isinstance(move[0], Card):
            last_card = move[-1]
            if last_card.value in ("wild", "wild_draw_four"):
                chosen_color = random.choice(
                    ["red", "blue", "yellow", "green"]
                )

        return move, chosen_color
