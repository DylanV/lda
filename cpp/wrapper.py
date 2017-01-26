from os.path import join, isfile, abspath
import os
from platform import system
from subprocess import Popen, PIPE


WINDOWS = True if system() == 'Windows' else False
WORK_DIR = join(os.getcwd(), 'build')
LDA_PATH = join(os.getcwd(), 'build', 'lda.exe') if WINDOWS else join(os.getcwd(), 'build', 'lda')
OUT_PATH = join('..','..','param','artificial')
CORPUS_PATH = join('..','..','datasets','artificial.dat')
assert isfile(LDA_PATH), ('Unable to find the lda binary in the '
                                  'build directory, have you forgotten to compile it?: {}'
                                  ).format(LDA_PATH)

def run_LDA():
    lda_args = (abspath(LDA_PATH),
            '-c', abspath(OUT_PATH),
            '-t', '10',
            '-o', abspath(OUT_PATH),
            '-i', '3')

    p = Popen(lda_args, stdout=PIPE, stderr=PIPE)
    # p.wait()
    out, err = p.communicate()
    print(out)
    print(err)
    print(p.returncode)


def main(args):
    run_LDA()

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))