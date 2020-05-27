class Event:
    def __init__(self, delta, element, entity):
        self.delta = delta
        self.element = element
        self.entity = entity

    def execute(self):
        self.element.execute(entity)

    def __str__(self):
        return "(%s - %s - %s)" % (self.delta, self.element, self.entity)

class Controller:
    controller = None

    def __init__(self, sim_time):
        self.time = 0
        self.sim_time = sim_time
        self.events = []

        Controller.controller = self

    def register_event(self, event):
        self.events.append( event )

    def start_simulation(self):
        while( len(self.events) > 0 ):
            #print("at time %d" % self.time, [ event.__str__() for event in self.events] )
            #print()

            next_time = min( [event.delta for event in self.events] )
            self.time = next_time

            if self.time > self.sim_time:
                return

            for event in self.events:
                if event.delta == next_time:
                    self.events.remove(event)
                    event.element.execute(event.entity)
                    break

    def get_sim_time(self):
        return self.sim_time

    def get_controller():
        return Controller.controller
