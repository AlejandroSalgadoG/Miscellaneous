from Element import Element
from Controller import Controller, Event
from Functions import avg

class Queue(Element):
    def __init__(self, name):
        super().__init__(name)
        self.entities = []
        self.n_items = 0
        self.entrances = []
        self.exits = []

    def execute(self, entity):
        controller = Controller.get_controller()

        self.entities.append(entity)
        #print("entered %.2f kg, %d elements in queue %s at time %d" % (entity.get_kg(), len(self.entities), self.name, controller.time))

        controller.register_event( Event(controller.time, self.connections[0], self) )

        self.n_items += 1
        self.entrances.append( controller.time )

    def get_entity(self):
        if len(self.entities) == 0:
            return None
        self.exits.append( Controller.get_controller().time )
        return self.entities.pop(0)

    def calc_wait_times(self):
        wait_times = [exit - self.entrances[idx] for idx, exit in enumerate(self.exits)]
        if wait_times == []:
            wait_times = [0]
        return wait_times

    def get_info(self):
        wait_times = self.calc_wait_times()

        data = (self.name, self.n_items, len(self.entities), max(wait_times), min(wait_times), avg(wait_times))

        info =  "Queue: %s\n"
        info += "       Entered %d elements\n"
        info += "       It has %d elements\n"
        info += "       Max wait time %f\n"
        info += "       Min wait time %f\n"
        info += "       Avg wait time %f\n"
        #print(info % data)

        return data

    def __str__(self):
        return "%s %d" % (self.name, len(self.entities))
