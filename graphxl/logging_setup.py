"""
This module sets up the logging configuration for the workbench.
"""
import logging
from logging import getLogger

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
