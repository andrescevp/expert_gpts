import logging
from time import sleep

logger = logging.getLogger(__name__)


def retry(max: int = 10, backoff: int = 2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max:
                        logger.exception(
                            f"Error. Waiting {backoff ** (attempt + 2)} seconds..."
                        )
                        sleep(backoff ** (attempt + 2))
                    else:
                        raise e

        return wrapper

    return decorator
