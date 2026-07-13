from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mars import Mars
from .duke import DukeMTMCVidReID
from .market1501 import Market1501


__vidreid_factory = {
    'mars': Mars,
    'duke': DukeMTMCVidReID,
    'market1501': Market1501,
}


def get_names():
    return list(__vidreid_factory.keys())

def init_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)
