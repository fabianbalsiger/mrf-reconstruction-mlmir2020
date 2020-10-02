import argparse
import io
import os
import urllib.request as request
import zipfile


def main(url: str, data_dir: str):
    # download the data
    print(f'Downloading... ({url})')
    resp = request.urlopen(url)
    zip_ = zipfile.ZipFile(io.BytesIO(resp.read()))

    print(f'Extracting...')
    members = zip_.infolist()
    for member in members:
        if member.filename.endswith('.npy') or member.filename.endswith('.pkl'):
            if not os.path.basename(member.filename):
                # is a directory
                continue
            zip_.extract(member, data_dir)
            print(f'extract {os.path.join(data_dir, member.filename)}')
    print('Finished')


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Pull example data')

    parser.add_argument(
        '--url',
        type=str,
        default='https://github.com/fabianbalsiger/mrf-reconstruction-mlmir2020/releases/download/v1.0.0/example-data.zip',
        help='Path to the example data zip file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./../../',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.url, args.data_dir)
