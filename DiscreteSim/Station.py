from numpy.random import choice

from Element import Element
from Entity import Entity
from Controller import Controller, Event

class Station(Element):
    def __init__(self, name, distribution, parameters, transitions=None):
        super().__init__(name)
        self.busy_until = 0
        self.parameters = parameters
        self.distribution = distribution
        self.transitions = transitions
        self.kg_processed = 0

    def execute(self, queue):
        controller = Controller.get_controller()

        if self.busy_until > controller.time:
            #print("Ativivity %s is buisy at time %d" % (self.name, controller.time))
            controller.register_event( Event( self.busy_until, self, queue) )
            return

        entity = queue.get_entity()

        delta = self.distribution(*self.parameters) * entity.get_kg()
        next_element = choice(self.connections, p=self.transitions)
        self.busy_until = controller.time + delta

        #print("Processing %.2f kg in %s at time %d until time %d" % (entity.get_kg(), self.name, controller.time, self.busy_until))

        self.kg_processed += entity.get_kg()
        for next_element, probability in zip(self.connections, self.transitions):
            new_entity = Entity(entity.get_kg() * probability, entity.get_creation())
            controller.register_event( Event(self.busy_until, next_element, new_entity) )

    def get_info(self):
        data = (self.name, self.kg_processed)

        info =  "Station: %s\n"
        info += "         Processed %.2f kg\n"
        #print(info % data)

        return data

    def __str__(self):
        return "%s %s" % (self.name, self.busy_until)
