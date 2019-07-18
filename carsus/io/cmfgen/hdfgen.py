import os
import glob
from carsus import logger


def hdf_dump(cmfgen_dir, patterns, parser, chunk_size=10, ignore_patterns=[]):
    files = []
    ignore_patterns = ['.h5'] + ignore_patterns
    for case in patterns:
        path = '{0}/**/*{1}*'.format(cmfgen_dir, case)
        files = files + glob.glob(path, recursive=True)

        for i in ignore_patterns:
            files = [f for f in files if i not in f]

    n = chunk_size
    files_chunked = [files[i:i+n] for i in range(0, len(files), n)]

    # Divide read/dump in chunks for less I/O
    for chunk in files_chunked:

        _ = []
        for fname in chunk:
            output = fname.replace(cmfgen_dir, '')
            try:
                obj = parser.__class__(fname)
                logger.info('Parsed {}'.format(output))
                _.append(obj)

            except:
                logger.error('Failed parsing {}'.format(output))

        for obj in _:
            try:
                obj.to_hdf()
                logger.info('Dumped {}.h5'.format(output))

            except:
                logger.error('Failed dump {}'.format(output))
