import logging

from yachalk import chalk

from magnus.interaction import (end_interactive_session,
                                get_experiment_tracker_context,
                                get_from_catalog, get_object, get_parameter,
                                get_run_id, get_secret, put_in_catalog,
                                put_object, start_interactive_session,
                                store_parameter, track_this)
from magnus.sdk import AsIs, Pipeline, Task

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
