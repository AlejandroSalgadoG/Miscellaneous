from numpy.random import choice

from Element import Element
from Entity import Entity
from Data import avocato_weights
from Controller import Controller, Event

class PaletStation(Element):
    def __init__(self, name, distribution, parameters, capacity, transitions=None):
        super().__init__(name)
        self.weight = 0
        self.busy_until = 0
        self.n_completed = 0
        self.capacity = capacity
        self.parameters = parameters
        self.distribution = distribution
        self.transitions = transitions

    def execute(self, queue):
        controller = Controller.get_controller()

        if self.busy_until > controller.time:
            #print("Ativivity %s is buisy at time %d" % (self.name, controller.time))
            controller.register_event( Event( self.busy_until, self, queue) )
            return

        entity = queue.get_entity()
        self.weight += entity.get_kg()
        #print("There is %.2f kg in %s at time %d" % (self.weight, self.name, controller.time))

        if(self.weight >= self.capacity):
            delta = self.distribution(*self.parameters)
            next_element = choice(self.connections, p=self.transitions)
            self.busy_until = controller.time + delta

            self.weight = 0
            self.n_completed += 1
            entity = Entity(self.capacity, entity.get_creation())
            controller.register_event( Event(self.busy_until, next_element, entity) )

    def get_info(self):
        data = (self.name, self.n_completed)

        info =  "Station: %s\n"
        info += "         Processed %d entities\n"
        #print(info % data)

        return data

    def __str__(self):
        return "%s %s" % (self.name, self.busy_until)
