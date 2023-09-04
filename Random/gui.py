from __future__ import annotations
import wx

from items import Item, ItemList


def str2float(value):
    try:
        return float(value)
    except:
        return None


def str2int(value):
    try:
        return int(value)
    except:
        return None




class Entry:
    def __init__(self, frame: wx.Frame, x_pos: int, y_pos: int, item_list: ItemList):
        self.y_pos = y_pos
        self.item_list = item_list
        self.item = None

        self.name = wx.StaticText(frame, id=-1, label="")
        self.name.SetPosition((x_pos + 20, self.y_pos))

        self.quantity = wx.StaticText(frame, id=-1, label="0")
        self.quantity.SetPosition((x_pos + 130, self.y_pos))

        self.update = wx.Button(frame, wx.ID_CLOSE, "cambiar")
        self.update.SetPosition((x_pos + 160, self.y_pos))
        self.update.Bind(wx.EVT_BUTTON, self.update_quant)

        self.prob = wx.StaticText(frame, id=-1, label="0")
        self.prob.SetPosition((x_pos + 260, self.y_pos))

        self.fix = wx.Button(frame, wx.ID_CLOSE, "fijar")
        self.fix.SetPosition((x_pos + 340, self.y_pos))
        self.fix.Bind(wx.EVT_BUTTON, self.fix_prob)

        self.free = wx.Button(frame, wx.ID_CLOSE, "liberar")
        self.free.SetPosition((x_pos + 440, self.y_pos))
        self.free.Bind(wx.EVT_BUTTON, self.unfix_prob)

    def set_item(self, item: Item):
        self.item = item
        self.name.SetLabel(item.name)
        self.update_quantity()
    
    def update_prob(self):
        self.prob.SetLabel(f"{self.item.prob:.3%}")

    def update_quantity(self):
        self.quantity.SetLabel(f"{self.item.quantity}")

    def update_quant(self, e):
        dlg = wx.TextEntryDialog(None, 'Cantidad:', "", style=wx.OK)
        dlg.ShowModal()
        quantity_str = dlg.GetValue()
        quantity = str2int(quantity_str)
        if quantity is not None: 
            self.item_list.update_quantity(self.item, quantity)
        dlg.Destroy()

    def fix_prob(self, e):
        dlg = wx.TextEntryDialog(None, 'Probabilidad:', "", style=wx.OK)
        dlg.ShowModal()
        prob_str = dlg.GetValue()
        prob = str2float(prob_str)
        if prob is not None: 
            self.prob.SetForegroundColour((255, 0, 0))
            self.item_list.fix_prob(prob, self.item)
        dlg.Destroy()
        
    def unfix_prob(self, e):
        self.prob.SetForegroundColour((255, 255, 255))
        self.item_list.unfix_prob(self.item)


class PlaySection:
    def __init__(self, frame: wx.Frame, y_pos: int, item_list: ItemList):
        self.y_pos = y_pos
        self.item_list = item_list

        font = wx.Font(100, wx.FONTFAMILY_MODERN, 0, 90, underline = False,
     faceName ="")

        self.button = wx.Button(frame, wx.ID_CLOSE, "jugar")
        self.button.SetPosition((1100, self.y_pos))
        self.button.SetFont(font)
        self.button.SetSize((530, 170))
        self.button.SetBackgroundColour((139, 0, 0))
        self.button.Bind(wx.EVT_BUTTON, self.play)

    def play(self, e) -> None:
        item = self.item_list.get_item()
        wx.MessageBox(item.name, 'Info')
