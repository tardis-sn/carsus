import glob
import logging

logger = logging.getLogger(__name__)

def hdf_dump(cmfgen_dir, patterns, parser, chunk_size=10, ignore_patterns=[]):
    """Function to parse and dump the entire CMFGEN database.

    Parameters
    ----------
    cmfgen_dir : path
        Path to the CMFGEN atomic database
    patterns : list of str
        String patterns to search for
    parser : class
        CMFGEN parser class
    chunk_size : int, optional
        Number of files to parse together, by default 10
    ignore_patterns : list, optional
        String patterns to ignore, by default []
    """
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
            try:
                obj = parser.__class__(fname)
                logger.info('Parsed {}'.format(fname))
                _.append(obj)

            except TypeError:
                logger.error('Failed parsing {} (try checking `find_row` function)'.format(fname))

            except UnboundLocalError:
                logger.error('Failed parsing {} (try checking `to_float` function)'.format(fname))

            except IsADirectoryError:
                logger.error('Failed parsing {} (is a directory)'.format(fname))

        for obj in _:
            obj.to_hdf()
            logger.info('Dumped {}.h5'.format(obj.fname))
