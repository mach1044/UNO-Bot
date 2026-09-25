from engine.Bot import Bot
from engine.Card import Card


class StrongerHeuristicBot(Bot):
    COLORS = (
        "red",
        "blue",
        "yellow",
        "green",
    )

    def choose_move(
        self,
        observation: dict,
        legal_moves: list[list[object]],
    ) -> tuple[list[object], str | None]:
        if not legal_moves:
            raise ValueError("The bot was given no legal moves")

        card_moves = [
            move
            for move in legal_moves
            if move and isinstance(move[0], Card)
        ]

        if not card_moves:
            return legal_moves[0], None

        if observation["pending_draw"] > 0:
            move = self._choose_pending_draw_move(card_moves)
            if move is None:
                return legal_moves[0], None

            return move, self._chosen_wild_color(observation, move)

        candidates = card_moves

        # Coloured moves always take priority over either kind of wild.
        colored_moves = [
            move
            for move in candidates
            if move[-1].color is not None
        ]
        if colored_moves:
            candidates = colored_moves

        # Protect numbered values represented in multiple colours. Playing a
        # duplicate of the same colour does not damage a cross-colour match.
        protected_values = self._cross_color_number_values(observation)
        preservation_costs = {
            id(move): int(
                move[0].value.isdigit()
                and int(move[0].value) in protected_values
            )
            for move in candidates
        }
        lowest_cost = min(preservation_costs.values())
        candidates = [
            move
            for move in candidates
            if preservation_costs[id(move)] == lowest_cost
        ]

        # Prefer the move whose final colour is strongest after playing it.
        color_strengths = {
            id(move): self._resulting_color_strength(observation, move)
            for move in candidates
        }
        highest_color_strength = max(color_strengths.values())
        candidates = [
            move
            for move in candidates
            if color_strengths[id(move)] == highest_color_strength
        ]

        # This rule intentionally runs after colour strength. Draw cards are
        # conserved, while skips and reverses are spent first.
        action_priorities = {
            id(move): self._action_priority(move[-1])
            for move in candidates
        }
        highest_action_priority = max(action_priorities.values())
        candidates = [
            move
            for move in candidates
            if action_priorities[id(move)] == highest_action_priority
        ]

        # If every requested heuristic ties, shedding more cards is beneficial.
        move = max(candidates, key=len)
        return move, self._chosen_wild_color(observation, move)

    @staticmethod
    def _choose_pending_draw_move(
        card_moves: list[list[Card]],
    ) -> list[Card] | None:
        draw_two_moves = [
            move
            for move in card_moves
            if move[0].value == "draw_two"
        ]
        if draw_two_moves:
            return max(draw_two_moves, key=len)

        wild_draw_four_moves = [
            move
            for move in card_moves
            if move[0].value == "wild_draw_four"
        ]
        if wild_draw_four_moves:
            return max(wild_draw_four_moves, key=len)

        return None

    @staticmethod
    def _cross_color_number_values(observation: dict) -> set[int]:
        hand_counts = observation["hand_counts"]

        return {
            value_index
            for value_index in range(10)
            if sum(
                hand_counts[color_index][value_index] > 0
                for color_index in range(4)
            ) > 1
        }

    @staticmethod
    def _remaining_color_counts(
        observation: dict,
        move: list[Card],
    ) -> list[int]:
        remaining_counts = list(observation["color_distribution"][:4])

        for card in move:
            if card.color is not None:
                remaining_counts[card.color_index] -= 1

        return remaining_counts

    def _resulting_color_strength(
        self,
        observation: dict,
        move: list[Card],
    ) -> int:
        remaining_counts = self._remaining_color_counts(observation, move)
        final_card = move[-1]

        if final_card.color is None:
            return max(remaining_counts)

        return remaining_counts[final_card.color_index]

    @staticmethod
    def _action_priority(card: Card) -> int:
        if card.value in ("skip", "reverse"):
            return 2
        if card.value in ("draw_two", "wild_draw_four"):
            return 0
        return 1

    def _chosen_wild_color(
        self,
        observation: dict,
        move: list[Card],
    ) -> str | None:
        if move[-1].color is not None:
            return None

        remaining_counts = self._remaining_color_counts(observation, move)
        strongest_color_index = max(
            range(len(self.COLORS)),
            key=lambda index: remaining_counts[index],
        )
        return self.COLORS[strongest_color_index]
