import collections.abc
import importlib

from ml_logger import logger
import sys
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)
sys.path.insert(0, parent_dir)

def import_class(_class):
    if type(_class) is not str:
        return _class
    
    repo_name = __name__.split(".")[0]
    
    module_name = ".".join(_class.split(".")[:-1])
    
    class_name = _class.split(".")[-1]
    
    module = importlib.import_module(f"{repo_name}.{module_name}")
    
    _class = getattr(module, class_name)
    print(f"[ utils/config ] Imported {repo_name}.{module_name}:{class_name}")
    return _class


class Config(collections.abc.Mapping):
    def __init__(self, _class, print_info=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = {}

        for key, val in kwargs.items():
            self._dict[key] = val

        if print_info:
            print(self)

        if savepath is not None:
            logger.save_pkl(self, savepath)
            print(f"[ utils/config ] Saved config to: {savepath}\n")

    def __repr__(self):
        string = f"\n[utils/config ] Config: {self._class}\n"
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f"    {key}: {val}\n"
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        if attr == "_dict" and "_dict" not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        return instance
