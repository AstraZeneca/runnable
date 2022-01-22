
import logging

from yachalk import chalk

from magnus.interaction import (get_from_catalog, get_parameter, get_secret,
                                put_in_catalog, store_parameter, track_this)

chalk_colors = {
    'debug': chalk.grey,
    'info': chalk.green,
    'warning': chalk.yellow_bright,
    'error': chalk.bold.red
}


class ColorFormatter(logging.Formatter):
    """
    Custom class to get colors to logs
    """

    def __init__(self, *args, **kwargs):
        # can't do super(...) here because Formatter is an old school class
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        levelname = record.levelname
        color = chalk_colors[levelname.lower()]
        message = logging.Formatter.format(self, record)
        return color(message)


logging.ColorFormatter = ColorFormatter  # type: ignore
