from Deck import Deck
from Card import Card

class Player:
    def __init__(self):
        self.hand = []
        # red = 0, blue = 1, yellow = 2 , green = 3, none = 4
        self.color_dist = [[],[],[],[],[]]
        # 0-9 for numbers, skip = 10, reverse = 11, draw2 = 12, +4 = 14, colour change = 15
        self.val_dist = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    def draw_card(self, deck: Deck):
        card = deck.deal_card()
        if card.color == "red":
            color_index = 0
        elif card.color == "blue":
            color_index = 1
        elif card.color == "yellow":
            color_index = 2
        elif card.color == "green":
            color_index = 3
        else:
            color_index = 4

        if card.value.isdigit():
            value_index = int(card.value)
        elif card.value == "skip":
            value_index = 10
        elif card.value == "reverse":
            value_index = 11
        elif card.value == "draw_two":
            value_index = 12
        elif card.value == "wild":
            value_index = 13
        elif card.value == "wild_draw_four":
            value_index = 14

        self.hand.append(card)
        self.color_dist[color_index].append(card)
        self.val_dist[value_index].append(card)

    def remove_card(self, card: Card):
        if card.color == "red":
            color_index = 0
        elif card.color == "blue":
            color_index = 1
        elif card.color == "yellow":
            color_index = 2
        elif card.color == "green":
            color_index = 3
        else:
            color_index = 4

        if card.value.isdigit():
            value_index = int(card.value)
        elif card.value == "skip":
            value_index = 10
        elif card.value == "reverse":
            value_index = 11
        elif card.value == "draw_two":
            value_index = 12
        elif card.value == "wild":
            value_index = 13
        elif card.value == "wild_draw_four":
            value_index = 14

        self.hand.remove(card)
        self.color_dist[color_index].remove(card)
        self.val_dist[value_index].remove(card)
