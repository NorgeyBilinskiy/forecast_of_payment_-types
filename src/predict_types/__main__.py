import os
import subprocess
from loguru import logger

def run_scripts():
    os.chdir(os.path.dirname(__file__))
    scripts = ['stacking.py']
    for script in scripts:
        try:
            subprocess.run(['python', script], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f'Erorr run {script}: {e}')

if __name__ == '__main__':
    logger.info('START PREDICTION OF TYPES')
    run_scripts()
    logger.info('END PREDICTION OF TYPES')