import wx

from items import Item, ItemList
from gui import Entry, PlaySection

class Builder:
    @staticmethod
    def build(raw_item_list: dict[str, int]):
        frame = wx.Frame(None, title="Juego", size=(1700,1050))
        x_pos = 0

        item_list = ItemList()
        for idx, (name, quantity) in enumerate(raw_item_list.items(), start=1):

            if 39 <= idx < 78:
                x_pos = 550
                idx -= 38
            elif idx >= 78:
                x_pos = 1100
                idx -= 77

            entry = Entry(frame, x_pos, 25*idx, item_list)
            item = Item(entry, name, quantity)
            entry.set_item(item)
            item_list.add_item(item)

        PlaySection(frame, 675, item_list)

        return frame
