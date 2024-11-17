import os
import subprocess
from loguru import logger

def run_module_scripts():
    os.chdir(os.path.dirname(__file__))
    submodules = [
        'src',
    ]
    for module in submodules:
        script_path = os.path.join(os.getcwd(), module, '__main__.py')
        try:
            subprocess.run(['python', script_path], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f'Error during execution file {script_path}: {e}')


if __name__ == '__main__':
    run_module_scripts()
