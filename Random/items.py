import numpy as np


class Item:
    def __init__(self, entry, name: str, quantity: int):
        self.entry = entry
        self.name = name
        self.quantity = quantity
        self.fixed = False
        self.prob = 0

    def set_prob(self, prob: float) -> None:
        self.prob = prob
        self.entry.update_prob()

    def set_quantity(self, quantity: int) -> None:
        self.quantity = quantity
        self.entry.update_quantity()

    def fix_prob(self, prob: float) -> None:
        self.fixed = True
        self.set_prob(prob)

    def unfix_prob(self) -> None:
        self.fixed = False
        self.set_prob(0)


class ItemList:
    def __init__(self):
        self.items = []
        self.n_items = 0

    def get_total_non_fixed(self) -> int:
        return sum(item.quantity for item in self.items if not item.fixed)

    def get_fixed_prob(self) -> float:
        return sum(item.prob for item in self.items if item.fixed)

    def get_non_fixed_prob(self) -> float:
        return 1 - self.get_fixed_prob()

    def get_probs(self) -> list[float]:
        return [item.prob for item in self.items]

    def update_probs(self) -> None:
        non_fixed_prob = self.get_non_fixed_prob()
        total_non_fixed = self.get_total_non_fixed()

        for item in self.items:
            if not item.fixed:
                item.set_prob( item.quantity * non_fixed_prob / total_non_fixed )

    def add_item(self, item: Item) -> None:
        self.items.append(item)
        self.n_items += 1
        self.update_probs()

    def fix_prob(self, prob: float, item: Item) -> None:
        if 0 <= prob <= 1:
            item.fix_prob(prob)
            self.update_probs()

    def unfix_prob(self, item: Item) -> None:
        item.unfix_prob()
        self.update_probs()

    def update_quantity(self, item: Item, quantity: int):
        if quantity >= 0:
            item.set_quantity( quantity )
            self.update_probs()
            self.save()

    def get_item(self) -> Item:
        [item_idx] = np.random.choice(range(self.n_items), 1, p=self.get_probs())
        item = self.items[item_idx]
        self.update_quantity(item, item.quantity - 1)
        self.update_probs()
        return item

    def print_probs(self) -> None:
        for item in self.items:
            print(item.name, item.quantity, f"{item.prob:.2f}")

    def save(self) -> None:
        with open("data_bk.py", "w") as f:
            f.write("raw_items = {\n")
            for item in self.items:
                f.write(f'"{item.name}": {item.quantity},\n')
            f.write("}")
