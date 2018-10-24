import yaml


class DotDict(dict):
    """A dictionary where you can use the dot syntax to access and set items
    and load a yaml file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def load_yaml(self, filename):
        with open(filename) as fp:
            self.__init__(**yaml.load(fp))

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        v = DotDict(value) if isinstance(value, dict) else value
        self.__setitem__(key, v)

    def __setitem__(self, key, value):
        v = DotDict(value) if isinstance(value, dict) else value
        super().__setitem__(key, v)
        self.__dict__.update({key: v})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]
