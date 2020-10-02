import time


def get_timestamp() -> str:
    return time.strftime('%y%m%d-%H%M%S')
