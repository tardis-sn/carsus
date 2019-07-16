import os
import glob

class Colors:
    OK = '\033[92m'
    FAIL = '\033[91m'
    END = '\033[0m'

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
            print('Parsing file -- {}'.format(fname), end=' ')
            try:
                obj = parser.__class__(fname)
                print(Colors.OK + 'OK' + Colors.END)
                _.append(obj)

            except:
                print(Colors.FAIL + 'FAIL' + Colors.END)

        for obj in _:
            print('Dumping file -- {}.h5'.format(obj.fname), end=' ')
            try:
                obj.to_hdf()
                print(Colors.OK + 'OK' + Colors.END)

            except:
                print(Colors.FAIL + 'FAIL' + Colors.END)