import os
from sys import argv

if __name__ == '__main__':
    if len(argv) < 2:
        print('Command not supplied')
        print('python docker.py build')
        print('python docker.py run')
        print('python docker.py run_safe # (does not mount the host copy of lcbg dir)')

    elif argv[1] == 'build' or argv[1] == '-b' or argv[1] == 'b':
        os.system('docker build -t lcbg .')

    elif argv[1] == 'run' or argv[1] == '-r' or argv[1] == 'r':
        path_to_this_dir = os.path.dirname(os.path.abspath(__file__))
        os.system("docker run --mount type=bind,src={},dst=/home/jovyan/lcbg -it -p 8888:8888 lcbg".format(path_to_this_dir))

    elif argv[1] == 'run_safe':
        os.system('docker run -it -p 8888:8888 lcbg')
