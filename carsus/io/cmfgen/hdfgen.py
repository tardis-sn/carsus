import os
import glob
import argparse
import warnings
from carsus.io.cmfgen import (CMFGENEnergyLevelsParser,
                              CMFGENOscillatorStrengthsParser,
                              CMFGENCollisionalDataParser,
                              CMFGENPhotoionizationCrossSectionParser)

CHUNK_SIZE = 10
osc_patterns = ['osc', 'OSC', 'Osc']
col_patterns = ['col', 'COL', 'Col']
pho_patterns = ['pho', 'PHO', 'Pho']

class Colors:
    OK = '\033[92m'
    FAIL = '\033[91m'
    END = '\033[0m'

def dump(patterns, cmfgenparser, chunk_size=CHUNK_SIZE):
    files = []
    for case in patterns:
        path = CMFGEN_DIR + '/**/*{}*'.format(case)
        files = files + glob.glob(path.replace('//', '/'), recursive=True)
        files = [ f for f in files if not f.endswith('.h5') ]

    n = chunk_size
    files_chunked = [ files[i:i+n] for i in range(0, len(files), n) ]

    # Divide read/dump in chunks for less I/O
    for chunk in files_chunked:
        
        _ = []
        for fname in chunk:
            print('Parsing file -- {}'.format(fname), end=' ')

            try:
                obj = cmfgenparser.__class__(fname)
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


CMFGEN_DIR = os.getenv('CMFGEN_DIR')
if not CMFGEN_DIR:
    raise Exception('`CMFGEN_DIR` variable not set')

if not os.path.isdir(CMFGEN_DIR):
    raise Exception('`CMFGEN_DIR` variable is set but folder doesn\'t exist')



def main():
    optparser = argparse.ArgumentParser(description='Script to parse and dump CMFGEN Atomic Database.')   
    optparser.add_argument('-W', '--warnings', action='store_true', default=True, dest='disable_warnings',
                        help='no disable warnings')
    optparser.add_argument('-O', '--osc', action='store_true', default=False, dest='parse_osc',
                        help='parse OSC files')
    optparser.add_argument('-C', '--col', action='store_true', default=False, dest='parse_col',
                        help='parse COL files')
    optparser.add_argument('-P', '--PHO', action='store_true', default=False, dest='parse_pho',
                        help='parse PHO files')
    results = optparser.parse_args()
    
    if results.disable_warnings:
        warnings.simplefilter("ignore")
    
    if results.parse_osc:
        dump(osc_patterns, CMFGENEnergyLevelsParser())
        dump(osc_patterns, CMFGENOscillatorStrengthsParser())
    
    if results.parse_col:
        dump(col_patterns, CMFGENCollisionalDataParser())
    
    if results.parse_pho:
        dump(pho_patterns, CMFGENPhotoionizationCrossSectionParser())