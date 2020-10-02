import os


def get_checkpoint_path(checkpoint_dir: str, epoch: int, best_in_metric: str = None):
    if best_in_metric is not None:
        return os.path.join(checkpoint_dir, f'checkpoint_best-{best_in_metric}_ep{epoch:04}.pth')
    return os.path.join(checkpoint_dir, f'checkpoint_ep{epoch:04}.pth')


def prepare_epoch_result_directory(result_dir: str, epoch: int) -> str:
    """Creates a result directory named by the epoch on the filesystem.

    Args:
        result_dir (str): The root result directory.
        epoch (int): The epoch number.

    Returns:
        str: The path to the epoch result directory.
    """
    epoch_result_dir = os.path.join(result_dir, 'epoch_{:03d}'.format(epoch))
    os.makedirs(epoch_result_dir, exist_ok=True)
    return epoch_result_dir
