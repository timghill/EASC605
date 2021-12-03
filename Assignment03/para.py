class Para:
    def __init__(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

x = Para(ct=1, cw=3)

print(x)

print(x.ct)
