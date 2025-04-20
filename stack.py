class Stack:

    def __init__(self):
        self.history = []

    def append(self, operation):
        self.history.append(operation)

    def __str__(self):
        string = ""
        last_elem = self.history.pop()
        for hist in reversed(self.history):
            string += hist + "\n"
        string += last_elem
        return string
    
    def __repr__(self):
        return self.__str__()