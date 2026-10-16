"""Extract the official Zenodo repo zip and print a file inventory."""
import os
import sys
import zipfile

ZIP = r'_official\DSC-AMCNet-repo.zip'
DEST = r'_official\repo'

z = zipfile.ZipFile(ZIP)
bad = z.testzip()
print('zip CRC test:', 'OK' if bad is None else f'BAD entry {bad}')
z.extractall(DEST)
for root, _dirs, files in os.walk(DEST):
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, DEST)
        print(f'{os.path.getsize(p):>9,}  {rel}')
print('extracted to', os.path.abspath(DEST))
