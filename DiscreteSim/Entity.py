class Entity:
    def __init__(self, kg_avocato, time):
        self.kg_avocato = kg_avocato
        self.creation = time

    def get_creation(self):
        return self.creation

    def get_kg(self):
        return self.kg_avocato

    def __str__(self):
        return "Entity with %d kg" % self.kg_avocato
