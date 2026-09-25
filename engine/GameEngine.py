from .Deck import Deck
from .Card import Card
from .Bot import Bot
from .Permute import Permute
from .Player import Player
from collections import deque


#responsible for managing the game state
#responsible for running the game
class GameEngine:
    def __init__(self):
        self.restart_game()

    def restart_game(self):
        self.deck = Deck()
        self.deck.shuffle()

        self.player1 = Player()
        self.player2 = Player()
        self.discard_pile = []

        self.upcard = self.deck.deal_card()
        self.move_history = {
            self.player1: deque(maxlen=3),
            self.player2: deque(maxlen=3),
        }

        # force the upcard to be normal
        while self.upcard.value == "wild_draw_four" or self.upcard.value == "wild":
            self.deck.cards.append(self.upcard)
            self.deck.cardsLeft += 1
            self.deck.shuffle()
            self.upcard = self.deck.deal_card()

        self.current_player = self.player1
        self.pending_effect = None
        self.pending_draw = 0

        self.current_color = self.upcard.color

        self.winner = None
        self.game_over = False

        for _ in range(7):
            self.draw_card(self.player1)
            self.draw_card(self.player2)

    def draw_card(self, player: Player):
        if not self.deck.cards:
            if not self.discard_pile:
                raise RuntimeError("No cards are available to draw")

            self.deck.cards.extend(self.discard_pile)
            self.discard_pile.clear()
            self.deck.cardsLeft = len(self.deck.cards)
            self.deck.shuffle()

        return player.draw_card(self.deck)

    # deals with the legal card moves
    def legal_moves(self, player: Player, effect: str | None | Card) -> list[list[object]]:
        if effect is not None:
            if isinstance(effect, Card):
                drawn_card = effect
                moves = [["pass"]]
                additional_moves = []

                # A post-draw move must contain the exact card that was just
                # drawn. Its first card must still be legal against the
                # current colour/upcard; every later card has the same value.
                possible_first_cards = [
                    card
                    for card in player.val_dist[drawn_card.value_index]
                    if (
                        card.color == self.current_color
                        or card.value == self.upcard.value
                        or card.value in ("wild", "wild_draw_four")
                    )
                ]

                for first_card in possible_first_cards:
                    permutations = (
                        Permute().permute_possible_moves_given_first_card(
                            player,
                            first_card,
                        )
                    )
                    additional_moves.extend(
                        move
                        for move in permutations
                        if drawn_card in move
                    )

                moves.extend(additional_moves)
                return moves
            elif effect == "skip":
                return [["pass"]]
            elif effect == "reverse":
                return [["pass"]]
            elif effect == "draw":
                moves = [["draw"]]
                if self.upcard.value == "draw_two":
                    additional_moves = []
                    for card in player.val_dist[12]:
                        additional_moves.extend(
                            Permute().permute_possible_moves_given_first_card(player, card)
                        )
                    for card in player.val_dist[14]:
                        additional_moves.extend(
                            Permute().permute_possible_moves_given_first_card(player, card)
                        )
                    moves.extend(additional_moves)
                elif self.upcard.value == "wild_draw_four":
                    additional_moves = []
                    for card in player.val_dist[14]:
                        additional_moves.extend(
                            Permute().permute_possible_moves_given_first_card(player, card)
                        )
                    moves.extend(additional_moves)
        else:
            moves = []
            moves.append(["draw_one"])

            if self.current_color == "red":
                color_index = 0
            elif self.current_color == "blue":
                color_index = 1
            elif self.current_color == "yellow":
                color_index = 2
            elif self.current_color == "green":
                color_index = 3
            else:
                color_index = 4

            if self.upcard.value.isdigit():
                value_index = int(self.upcard.value)
            else:
                value_index = {
                    "skip": 10,
                    "reverse": 11,
                    "draw_two": 12,
                    "wild": 13,
                    "wild_draw_four": 14,
                }[self.upcard.value]

            # A first card may match the current colour, the upcard's value,
            # or be either kind of wild. Do not add a card twice if it matches
            # both the colour and value.
            possible_first_cards = []
            for card in (
                player.color_dist[color_index]
                + player.val_dist[value_index]
                + player.val_dist[13]
                + player.val_dist[14]
            ):
                if card not in possible_first_cards:
                    possible_first_cards.append(card)

            for card in possible_first_cards:
                moves.extend(Permute().permute_possible_moves_given_first_card(player, card))

        return moves

    # Tracks player identity, hand size, colour/value distributions, full
    # colour-by-value hand counts, opponent hand size, upcard, current colour,
    # pending effect/draw count, and remaining deck size.
    def get_observation(self, player: Player) -> dict:

        if player is self.player1:
            opponent = self.player2
            player_id = 0
        else:
            opponent = self.player1
            player_id = 1

        hand_counts = [[0] * 15 for _ in range(5)]
        for card in player.hand:
            hand_counts[card.color_index][card.value_index] += 1

        return {
            "player_id": player_id,
            "hand_size": len(player.hand),
            "color_distribution": tuple(
                len(cards) for cards in player.color_dist
            ),
            "value_distribution": tuple(
                len(cards) for cards in player.val_dist
            ),
            "hand_counts": tuple(
                tuple(row) for row in hand_counts
            ),
            "opponent_hand_size": len(opponent.hand),
            "upcard_value": self.upcard.value,
            "upcard_color": self.upcard.color,
            "current_color": self.current_color,
            "pending_effect": self.pending_effect,
            "pending_draw": self.pending_draw,
            "deck_size": self.deck.cards_left(),
            "last_three_moves": self.move_history,
        }

    def execute_move(self, player: Player, move: list[object], chosen_color: str | None) -> None:
        if move[0] == "pass":
            self.pending_effect = None
            return None
        elif move[0] == "draw":
            while self.pending_draw > 0:
                self.pending_draw -= 1
                self.draw_card(player)
            self.pending_effect = None
            return None
        elif move[0] == "draw_one":
            return self.draw_card(player)
        else:
            for card in move:
                player.remove_card(card)

            # The final card becomes the upcard, and its colour becomes the
            # active colour. Wild colours are chosen separately by the bot.
            last_card = move[-1]
            self.discard_pile.append(self.upcard)
            self.discard_pile.extend(move[:-1])
            self.upcard = last_card

            if last_card.value in ("wild", "wild_draw_four"):
                if chosen_color not in ("red", "blue", "yellow", "green"):
                    raise ValueError("A wild move requires a chosen color")
                self.current_color = chosen_color
            else:
                self.current_color = last_card.color

            # Every card in a multi-card move has the same value, so the final
            # card identifies the effect and len(move) gives the stack size.
            if last_card.value == "draw_two":
                self.pending_effect = "draw"
                self.pending_draw += 2 * len(move)
            elif last_card.value == "wild_draw_four":
                self.pending_effect = "draw"
                self.pending_draw += 4 * len(move)
            elif last_card.value == "skip":
                self.pending_effect = "skip"
                self.pending_draw = 0
            elif last_card.value == "reverse":
                self.pending_effect = "reverse"
                self.pending_draw = 0
            else:
                self.pending_effect = None
                self.pending_draw = 0

            return None

    # walk the player through the process of making a move
    def play_turn(
        self,
        move: list[object],
        chosen_color: str | None,
        legal_moves: list[list[object]] | None = None,
    ) -> Player | None:
        """Apply one turn and return the winner, if the game has ended."""
        if self.game_over:
            raise RuntimeError("The game is already over")

        player = self.current_player
        if legal_moves is None:
            legal_moves = self.legal_moves(player, self.pending_effect)
        if move not in legal_moves:
            raise ValueError("Bot selected an illegal move")

        ifCard = self.execute_move(player, move, chosen_color)
        if isinstance(ifCard, Card):
            return ifCard
        elif move[0] == "draw":
            return None

        if not player.hand:
            self.winner = player
            self.game_over = True
            return player

        self.current_player = (
            self.player2 if player is self.player1 else self.player1
        )
        return None

    def game(
        self,
        player1_bot: Bot,
        player2_bot: Bot,
        max_turns: int | None = None,
    ) -> Player | None:
        if max_turns is not None and max_turns <= 0:
            raise ValueError("max_turns must be positive or None")

        bots = {
            self.player1: player1_bot,
            self.player2: player2_bot,
        }

        turns = 0
        while not self.game_over:
            if max_turns is not None and turns >= max_turns:
                return None

            player = self.current_player
            bot = bots[player]
            observation = self.get_observation(player)
            legal_moves = self.legal_moves(player, self.pending_effect)
            move, chosen_color = bot.choose_move(observation, legal_moves)

            ifCard = self.play_turn(move, chosen_color, legal_moves)

            if move[0] == "draw":
                observation = self.get_observation(player)
                legal_moves = self.legal_moves(player, None)
                move1, chosen_color = bot.choose_move(observation, legal_moves)
                ifCard = self.play_turn(move1, chosen_color, legal_moves)
                move.extend(move1)

            if isinstance(ifCard, Card):
                observation = self.get_observation(player) # is this observation correct???
                legal_moves = self.legal_moves(player, ifCard)
                move1, chosen_color = bot.choose_move(observation, legal_moves)
                ifCard = self.play_turn(move1, chosen_color, legal_moves)
                move.extend(move1)

            elif ifCard is not None:
                break

            self.move_history[player].append(move)
            if len(self.move_history[player]) > 3:
                self.move_history[player].popleft()

            turns += 1

        return self.winner
