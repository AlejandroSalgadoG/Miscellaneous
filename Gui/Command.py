from abc import ABC, abstractmethod

class Command(ABC):

    @abstractmethod
    def execute(self):
        pass

class NewCommand(Command):
    def execute(self):
        print("This is the new command")
