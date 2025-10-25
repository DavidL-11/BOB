import logging

"""
Custom logger for the BOB module.
"""

logger = logging.getLogger("BOB")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger.info("Logger initialized for BOB module.")
