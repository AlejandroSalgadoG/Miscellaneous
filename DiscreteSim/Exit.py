from Element import Element
from Controller import Controller
from Queue import Queue
from Functions import avg

class Exit(Element):
    def __init__(self, name):
        super().__init__(name)
        self.kg_descarted = 0
        self.times_in_system = []

    def execute(self, element):
        controller = Controller.get_controller()

        if isinstance(element, Queue):
            entity = element.get_entity()
        else:
            entity = element

        self.kg_descarted += entity.get_kg()
        self.times_in_system.append(controller.time - entity.get_creation())

        #print("Finishing %.2f kg in %s at time %d" % (entity.get_kg(), self.name, controller.time))

    def get_info(self):
        data = (self.name, self.kg_descarted)
        info =  "Exit: %s\n"
        info += "      Fnished %.2f kg\n"
        #print(info % data)

        return data

    def __str__(self):
        return "%s" % self.name
