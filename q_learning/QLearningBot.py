from engine.Bot import Bot
from engine.Card import Card
import torch
import random

class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(157, 128),
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

class QLearningBot(Bot):
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
        #
        hand_counts = observation["hand_counts"]
        # add the colors to the state (4 colours 13 possible values each + 2 wild cards) (0-53)
        for i in range(4):
            color_row = hand_counts[i]
            for j in range(13):
                state.append(color_row[j])
        state.append(hand_counts[4][13])
        state.append(hand_counts[4][14])
        # Encode the upcard value as one of 15 possible values. (54-68)
        upcard_value = observation["upcard_value"]
        if upcard_value.isdigit():
            upcard_value_index = int(upcard_value)
        else:
            upcard_value_index = Card.VALUE_INDEX[upcard_value]

        upcard_value_encoding = [0] * 15
        upcard_value_encoding[upcard_value_index] = 1
        state.extend(upcard_value_encoding)
        # current active colour (not necessarily upcard color) (69-72)
        current_color = observation["current_color"]
        current_color_encoding = [0] * 4
        current_color_index = Card.COLOR_INDEX[current_color]
        current_color_encoding[current_color_index] = 1
        state.extend(current_color_encoding)
        # pending effect 73-76
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
        # pending_draw 77
        state.append(observation["pending_draw"]/10)
        # opponent hand size 78
        state.append(observation["opponent_hand_size"]/10)
        # your hand size 79
        state.append(observation["hand_size"]/10)

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
                # 80 - 133
                remaining_hand
                # 134
                + [new_hand_count]
                # 135 - 149
                + final_card_encoding
                # 150 - 153
                + resulting_color_encoding
                # 154 - 156
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
                cards_removed.append(13*card_color+card_val)

        #80-133
        remaining_hand = state[:54]

        for card in cards_removed:
            remaining_hand[card] -= 1

        # 134
        new_hand_count = state[79] - len(move)/10
        final_card = move[-1]

        # 135-149
        final_card_encoding = [0] * 15
        final_card_encoding[final_card.value_index] = 1

        # 150-153
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
            # 154-156
            + non_card_encoding
        )
        return encoded_move
