import wx

from builder import Builder
from data import raw_items


def main():
    app = wx.App(redirect=False)   # Error messages go to popup window
    Builder.build(raw_items).Show()
    app.MainLoop()


if __name__ == '__main__':
   main()
