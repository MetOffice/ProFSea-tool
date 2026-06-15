import logging

from rich.logging import RichHandler

# Tie logger to the "profsea" namespace
logger = logging.getLogger("profsea")

# Print by default
logger.setLevel(logging.INFO)

# Use Rich!
rich_handler = RichHandler(
    rich_tracebacks=True,
    show_time=True,
    show_path=False,
    markup=True,
    log_time_format="[%H:%M:%S]",
)

formatter = logging.Formatter("%(message)s")
rich_handler.setFormatter(formatter)
logger.addHandler(rich_handler)
logger.propagate = False
