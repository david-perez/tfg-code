import logging


def build_logger(filename):
    logging_handlers = [logging.FileHandler('logs/' + filename), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=logging_handlers)

    return logging
