class Mean:
    @staticmethod
    def collect(x, y):
        return x + y

    @staticmethod
    def aggregate(x, count):
        return x / count


def mean():
    return Mean()
