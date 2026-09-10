class Card:
    COLOR_INDEX = {
        "red": 0,
        "blue": 1,
        "yellow": 2,
        "green": 3,
        None: 4,
    }

    VALUE_INDEX = {
        "skip": 10,
        "reverse": 11,
        "draw_two": 12,
        "wild": 13,
        "wild_draw_four": 14,
    }

    def __init__(self, value, color):
        self.color = color
        self.value = value

    @property
    def color_index(self):
        return self.COLOR_INDEX[self.color]

    @property
    def value_index(self):
        if self.value.isdigit():
            return int(self.value)

        return self.VALUE_INDEX[self.value]
