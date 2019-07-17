import os
import glob
from carsus import logger


def h5dump(cmfgen_dir, patterns, parser, chunk_size=10):
    files = []
    for case in patterns:
        path = cmfgen_dir + '/**/*{}*'.format(case)
        files = files + glob.glob(path.replace('//', '/'), recursive=True)
        files = [f for f in files if not f.endswith('.h5')]

    n = chunk_size
    files_chunked = [files[i:i+n] for i in range(0, len(files), n)]

    # Divide read/dump in chunks for less I/O
    for chunk in files_chunked:

        _ = []
        for fname in chunk:
            try:
                obj = parser.__class__(fname)
                logger.info('Parsed {}'.format(fname.replace(cmfgen_dir + 'atomic/', '')))
                _.append(obj)

            except:
                logger.error('Failed parsing {}'.format(fname.replace(cmfgen_dir + 'atomic/', '')))

        for obj in _:
            try:
                obj.to_hdf()
                logger.info('Dumped {}.h5'.format(fname.replace(cmfgen_dir + 'atomic/', '')))

            except:
                logger.error('Failed dump {}'.format(fname.replace(cmfgen_dir + 'atomic/', '')))