from Card import Card
from Player import Player
from itertools import permutations


class Permute:
    def permute_possible_moves_given_first_card(self, player: Player,
                                                firstCard: Card) -> list[list[Card]]:
        possible_moves = [[firstCard]]

        # The first card is already in the move. Only the player's other cards
        # with the same value can be played after it.
        if firstCard.value.isdigit():
            value_index = int(firstCard.value)
        else:
            value_index = {
                "skip": 10,
                "reverse": 11,
                "draw_two": 12,
                "wild": 13,
                "wild_draw_four": 14,
            }[firstCard.value]

        remaining_cards = [
            card
            for card in player.val_dist[value_index]
            if card is not firstCard
        ]

        for length in range(1, len(remaining_cards) + 1):
            for permutation in permutations(remaining_cards, length):
                possible_moves.append([firstCard, *permutation])

        return possible_moves
