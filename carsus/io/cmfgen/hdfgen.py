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
        path = f'{cmfgen_dir}/**/*{case}*'
        files = files + glob.glob(path, recursive=True)

        for i in ignore_patterns:
            files = [f for f in files if i not in f]

    n = chunk_size
    files_chunked = [files[i:i+n] for i in range(0, len(files), n)]
    logger.info(f'{len(files)} files selected.')

    # Divide read/dump in chunks for less I/O
    for chunk in files_chunked:

        _ = []
        for fname in chunk:
            try:
                obj = parser.__class__(fname)
                _.append(obj)

            # tip: check `find_row`
            except TypeError:
                logger.warning(f'`TypeError` raised while parsing `{fname}`.')

            # tip: check `to_float`
            except UnboundLocalError:
                logger.warning(f'`UnboundLocalError` raised while parsing `{fname}`.')

            except IsADirectoryError:
                logger.warning(f'`{fname}` is a directory.')

        for obj in _:
            obj.to_hdf()

    logger.info('Finished.')

