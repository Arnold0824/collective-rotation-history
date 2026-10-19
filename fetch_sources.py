"""Download checksum-verified beetle source files directly from the original repository."""
import argparse
import csv
import hashlib
from pathlib import Path
import tempfile
import urllib.request


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(item, destination):
    target = (destination / item['relative_path']).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError('Source manifest contains an unsafe local path')
    expected_size = int(item['bytes'])
    if target.exists():
        if target.stat().st_size != expected_size or sha256(target) != item['sha256']:
            raise ValueError(f'Existing source file differs from the recorded dataset: {target}')
        print(f'Already verified: {item["relative_path"]}', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(item['download_url'], headers={'User-Agent': 'collective-rotation-history/1.0'})
    with tempfile.TemporaryDirectory(prefix='source-download-', dir=target.parent) as temporary:
        partial = Path(temporary) / 'download'
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != expected_size or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Downloaded content does not match the recorded source: {item["relative_path"]}')
        partial.replace(target)
    print(f'Verified: {item["relative_path"]} ({size:,} bytes)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--only', help='Optionally download one named file from the manifest')
    args = parser.parse_args()
    manifest = Path(__file__).resolve().with_name('SOURCE_FILES.csv')
    with manifest.open(encoding='utf-8', newline='') as stream:
        items = [row for row in csv.DictReader(stream)
                 if row['dataset'] == 'beetles'
                 and (not args.only or Path(row['relative_path']).name == args.only)]
    if not items:
        parser.error('No source files match the requested selection')
    print(f'{len(items)} source files; {sum(int(x["bytes"]) for x in items):,} bytes', flush=True)
    for item in items:
        download(item, args.destination)


if __name__ == '__main__':
    main()
