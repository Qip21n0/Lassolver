class Base:
    def __init__(self, M, N):
        self.M = M
        self.N = N

    def members(self):
        for key in vars(self):
            print("{}: {}".format(key, vars(self)[key]))
