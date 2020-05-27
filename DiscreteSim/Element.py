class Element:
    def __init__(self, name):
        self.name = name
        self.connections = []

    def connect_with(self, element):
        self.connections.append(element)

    def execute(self, entity):
        print("Abstract execute")

    def get_info(self):
        return self.name
