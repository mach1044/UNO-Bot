from engine.Bot import Bot
from engine.Card import Card
import torch
import random


class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # Final input layout:
            # 0-79 current state, 80-239 opponent-turn history,
            # 240-316 candidate move.
            torch.nn.Linear(317, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, states, moves):
        combined_inputs = torch.cat(
            (states, moves),
            dim=-1
        )

        return self.layers(combined_inputs).squeeze(-1)


class QLearningBotTrackMoves(Bot):
    def __init__(self, epsilon: float = 0.0):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )
        self.model = QNetwork()
        self.episode_decisions = []
        self.epsilon = epsilon

    def choose_move(
            self,
            observation: dict,
            legal_moves: list[list[object]],
    ) -> tuple[list[object], str | None]:
        """Choose a legal move and, for a wild, its active colour."""

        state = self.encode_observation(observation)
        candidates = []

        for move in legal_moves:
            is_wild_move = (
                    move
                    and isinstance(move[0], Card)
                    and move[-1].value in ("wild", "wild_draw_four")
            )

            if is_wild_move:
                possible_colors = (
                    "red",
                    "blue",
                    "green",
                    "yellow",
                )
            else:
                possible_colors = (None,)

            for color in possible_colors:
                encoded_move = self.encode_move(state, move, color)
                candidates.append((move, color, encoded_move))

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        move_tensors = torch.tensor(
            [candidate[2] for candidate in candidates],
            dtype=torch.float32,
        )

        state_tensors = state_tensor.unsqueeze(0).expand(
            len(candidates),
            -1,
        )

        if random.random() < self.epsilon:
            best_index = random.randrange(
                len(candidates)
            )
        else:
            self.model.eval()

            with torch.no_grad():
                scores = self.model(state_tensors, move_tensors)

            best_index = int(torch.argmax(scores).item())

        best_move, best_color, _ = candidates[best_index]

        selected_move_tensor = move_tensors[best_index]
        player_id = observation["player_id"]
        self.episode_decisions.append(
            (
                state_tensor.detach().clone(),
                selected_move_tensor.detach().clone(),
                player_id,
                move_tensors.detach().clone(),
            )
        )

        return best_move, best_color

    def encode_observation(self, observation: dict):
        state = []
        hand_counts = observation["hand_counts"]
        # Current state 0-53: hand counts for 52 coloured card types
        # plus the two wild-card types.
        for i in range(4):
            color_row = hand_counts[i]
            for j in range(13):
                state.append(color_row[j])
        state.append(hand_counts[4][13])
        state.append(hand_counts[4][14])
        # Current state 54-68: upcard value.
        upcard_value = observation["upcard_value"]
        if upcard_value.isdigit():
            upcard_value_index = int(upcard_value)
        else:
            upcard_value_index = Card.VALUE_INDEX[upcard_value]

        upcard_value_encoding = [0] * 15
        upcard_value_encoding[upcard_value_index] = 1
        state.extend(upcard_value_encoding)
        # Current state 69-72: active colour (not necessarily upcard colour).
        current_color = observation["current_color"]
        current_color_encoding = [0] * 4
        current_color_index = Card.COLOR_INDEX[current_color]
        current_color_encoding[current_color_index] = 1
        state.extend(current_color_encoding)
        # Current state 73-76: pending effect.
        effect_indexes = {
            None: 0,
            "skip": 1,
            "reverse": 2,
            "draw": 3,
        }
        pending_effect_encoding = [0] * 4
        effect_index = effect_indexes[observation["pending_effect"]]
        pending_effect_encoding[effect_index] = 1
        state.extend(pending_effect_encoding)
        # Current state 77: pending draw amount.
        state.append(observation["pending_draw"] / 10)
        # Current state 78: opponent hand size.
        state.append(observation["opponent_hand_size"] / 10)
        # Current state 79: own hand size.
        state.append(observation["hand_size"] / 10)

        # Each historical turn uses 80 local features:
        # 0-53 cards played, 54 draw amount, 55-58 resulting colour,
        # 59 pass/skip, 60 voluntary draw, 61-64 pre-turn colour,
        # and 65-79 pre-turn upcard value.
        # In the full state, the older and newer turns occupy 80-159 and
        # 160-239 respectively.
        last_2_moves = []
        move_history = observation.get("last_three_moves", ())

        # The current engine exposes both players' deques. Select the
        # opponent's deque while also accepting the preferred future format,
        # where the observation directly contains only the opponent's turns.
        if isinstance(move_history, dict):
            histories = list(move_history.values())
            opponent_index = 1 if observation["player_id"] == 0 else 0
            move_history = (
                histories[opponent_index]
                if opponent_index < len(histories)
                else ()
            )

        turns = list(move_history)[-2:]
        turns = [None] * (2 - len(turns)) + turns

        for turn in turns:
            encoded_turn = [0.0] * 80

            if turn is None:
                last_2_moves.extend(encoded_turn)
                continue

            if isinstance(turn, dict):
                cards = turn.get("cards", ())
                cards_drawn = turn.get("cards_drawn", 0)
                resulting_color = turn.get(
                    "resulting_color",
                    turn.get("chosen_color"),
                )
                passed = turn.get("passed", False)
                voluntary_draw = turn.get("voluntary_draw", False)
                pre_turn_color = turn.get("pre_turn_color")
                pre_turn_value = turn.get("pre_turn_value")
            else:
                # Backward compatibility with the engine's current raw move
                # lists. Information not present in that format remains zero.
                cards = [item for item in turn if isinstance(item, Card)]
                actions = {
                    item for item in turn if isinstance(item, str)
                }
                cards_drawn = 1 if "draw_one" in actions else 0
                resulting_color = cards[-1].color if cards else None
                passed = "pass" in actions
                voluntary_draw = "draw_one" in actions
                pre_turn_color = None
                pre_turn_value = None

            for card in cards:
                if isinstance(card, Card):
                    card_color = card.color
                    card_value = card.value
                elif isinstance(card, dict):
                    card_color = card.get("color")
                    card_value = card.get("value")
                else:
                    card_color, card_value = card

                if card_value == "wild":
                    card_index = 52
                elif card_value == "wild_draw_four":
                    card_index = 53
                else:
                    color_index = Card.COLOR_INDEX[card_color]
                    value_index = (
                        int(card_value)
                        if card_value.isdigit()
                        else Card.VALUE_INDEX[card_value]
                    )
                    card_index = 13 * color_index + value_index

                encoded_turn[card_index] += 1

            encoded_turn[54] = cards_drawn / 10

            if resulting_color in ("red", "blue", "yellow", "green"):
                encoded_turn[
                    55 + Card.COLOR_INDEX[resulting_color]
                ] = 1

            encoded_turn[59] = float(bool(passed))
            encoded_turn[60] = float(bool(voluntary_draw))

            if pre_turn_color in ("red", "blue", "yellow", "green"):
                encoded_turn[
                    61 + Card.COLOR_INDEX[pre_turn_color]
                ] = 1

            if pre_turn_value is not None:
                value_index = (
                    int(pre_turn_value)
                    if pre_turn_value.isdigit()
                    else Card.VALUE_INDEX[pre_turn_value]
                )
                encoded_turn[65 + value_index] = 1

            last_2_moves.extend(encoded_turn)

        state.extend(last_2_moves)
        return state

    def encode_move(
            self,
            state: list[object],
            move: list[object],
            color: str | None,
    ):
        if isinstance(move[0], str):
            non_card_indexes = {
                "pass": 0,
                "draw": 1,
                "draw_one": 2,
            }
            non_card_encoding = [0] * 3
            non_card_encoding[non_card_indexes[move[0]]] = 1

            # The drawn card is not known yet, so retain the known hand
            # composition. Only its total size can be updated here.
            remaining_hand = state[:54]
            new_hand_count = state[79]
            if move[0] == "draw":
                new_hand_count += state[77]
            elif move[0] == "draw_one":
                new_hand_count += 0.1

            # No card was played, and the active colour stays unchanged.
            final_card_encoding = [0] * 15
            resulting_color_encoding = state[69:73]

            encoded_move = (
                # Candidate 0-53; combined input 240-293.
                    remaining_hand
                    # Candidate 54; combined input 294.
                    + [new_hand_count]
                    # Candidate 55-69; combined input 295-309.
                    + final_card_encoding
                    # Candidate 70-73; combined input 310-313.
                    + resulting_color_encoding
                    # Candidate 74-76; combined input 314-316.
                    + non_card_encoding
            )
            return encoded_move

        cards_removed = []
        for i in range(len(move)):
            card = move[i]
            if card.value_index == 13:
                cards_removed.append(52)
            elif card.value_index == 14:
                cards_removed.append(53)
            else:
                card_color = card.color_index
                card_val = card.value_index
                cards_removed.append(13 * card_color + card_val)

        # Candidate 0-53; combined input 240-293.
        remaining_hand = state[:54]

        for card in cards_removed:
            remaining_hand[card] -= 1

        # Candidate 54; combined input 294.
        new_hand_count = state[79] - len(move) / 10
        final_card = move[-1]

        # Candidate 55-69; combined input 295-309.
        final_card_encoding = [0] * 15
        final_card_encoding[final_card.value_index] = 1

        # Candidate 70-73; combined input 310-313.
        resulting_color_encoding = [0] * 4
        if final_card.color is None:
            resulting_color = color
        else:
            resulting_color = final_card.color

        if resulting_color is None:
            raise ValueError("A colour must be chosen for a wild card")

        color_index = Card.COLOR_INDEX[resulting_color]
        resulting_color_encoding[color_index] = 1

        non_card_encoding = [0] * 3
        encoded_move = (
                remaining_hand
                + [new_hand_count]
                + final_card_encoding
                + resulting_color_encoding
        # Candidate 74-76; combined input 314-316.
                + non_card_encoding
        )
        return encoded_move
