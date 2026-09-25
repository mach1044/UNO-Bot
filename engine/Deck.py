from .Card import Card
import random

class Deck:
    def __init__(self):
        self.cards = []
        self.cardsLeft = 108

        colors = ["red", "yellow", "green", "blue"]

        for color in colors:
            # One 0 per color
            self.cards.append(Card("0", color))

            # Two copies of 1-9 per color
            for i in range(1, 10):
                self.cards.append(Card(str(i), color))
                self.cards.append(Card(str(i), color))

            # Two skips per color
            self.cards.append(Card("skip", color))
            self.cards.append(Card("skip", color))

            # Two reverses per color
            self.cards.append(Card("reverse", color))
            self.cards.append(Card("reverse", color))

            # Two draw-twos per color
            self.cards.append(Card("draw_two", color))
            self.cards.append(Card("draw_two", color))

        # Four wild cards
        for _ in range(4):
            self.cards.append(Card("wild", None))

        # Four wild draw-four cards
        for _ in range(4):
            self.cards.append(Card("wild_draw_four", None))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self) -> Card:
        cardDealt = self.cards.pop()
        self.cardsLeft -= 1
        return cardDealt

    def cards_left(self):
        return self.cardsLeft
