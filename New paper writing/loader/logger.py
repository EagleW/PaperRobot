import logging
import sys


def get_logger(name, level=logging.INFO, handler=sys.stdout, filename=None,
               formatter='%(asctime)s %(name)s %(levelname)s %(message)s'
               ):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename:
        file_handler = logging.FileHandler(filename, 'w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()

