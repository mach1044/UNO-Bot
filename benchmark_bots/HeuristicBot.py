from engine.Bot import Bot
from engine.Card import Card


class HeuristicBot(Bot):
    ACTION_PRIORITY = {
        "wild_draw_four": 4,
        "draw_two": 3,
        "skip": 2,
        "reverse": 1,
    }

    def choose_move(
        self,
        observation: dict,
        legal_moves: list[list[object]],
    ) -> tuple[list[object], str | None]:
        if not legal_moves:
            raise ValueError("The bot was given no legal moves")

        card_moves = [
            move for move in legal_moves
            if move and isinstance(move[0], Card)
        ]

        if not card_moves:
            return legal_moves[0], None

        # Preserve wild cards whenever a coloured card can be played.
        colored_moves = [
            move for move in card_moves
            if move[-1].value not in ("wild", "wild_draw_four")
        ]
        candidate_moves = colored_moves or card_moves

        color_distribution = observation["color_distribution"]

        def move_score(move: list[Card]) -> tuple[int, int, int]:
            last_card = move[-1]
            action_priority = self.ACTION_PRIORITY.get(last_card.value, 0)

            remaining_color_count = 0
            if last_card.color is not None:
                remaining_color_count = color_distribution[last_card.color_index]
                remaining_color_count -= sum(
                    card.color_index == last_card.color_index
                    for card in move
                )

            return len(move), action_priority, remaining_color_count

        move = max(candidate_moves, key=move_score)
        chosen_color = None

        if move[-1].value in ("wild", "wild_draw_four"):
            colors = ["red", "blue", "yellow", "green"]
            chosen_color_index = max(
                range(len(colors)),
                key=lambda index: color_distribution[index],
            )
            chosen_color = colors[chosen_color_index]

        return move, chosen_color
