from Entity import Entity
from Element import Element

from Controller import Controller, Event

class Entry(Element):
    def __init__(self, name, entry_dist, entry_param, size_dist, size_param):
        super().__init__(name)
        self.n_items = 0
        self.entry_dist = entry_dist
        self.entry_param = entry_param
        self.size_dist = size_dist
        self.size_param = size_param

    def execute(self, _):
        controller = Controller.get_controller()

        delta = self.entry_dist(*self.entry_param)
        time = controller.time + delta
        queue = self.connections[0]
        entity = Entity(self.size_dist(*self.size_param), controller.time + delta)

        #print("Producing %.2f at time %d" % (entity.get_kg(), controller.time) )

        controller.register_event( Event(time, queue, entity) )
        controller.register_event( Event(time, self, None) )

        if time <= controller.get_sim_time():
            self.n_items += 1

    def get_info(self):
        info =  "Entry: %s\n" % self.name
        info += "       produced %d elements\n" % self.n_items
        return info 

    def __str__(self):
        return "%s" % (self.name)
