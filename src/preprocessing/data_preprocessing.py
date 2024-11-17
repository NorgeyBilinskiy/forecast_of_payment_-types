import os
import sys
from loguru import logger

sys.path.append('.')
from class_preprocessing import text_preprocessor

path_to_data = os.path.join(os.getcwd(), '../../input_data/payments_main.tsv')
output_path = os.path.join(os.getcwd(), '../temporary_data', 'processed_data.tsv')


if __name__ == '__main__':

    try:
        logger.info('START PREPROCESSING payments_main')
        data_processor = text_preprocessor(path_to_data, output_path)
        data_processor.summation()
        logger.info('END PREPROCESSING payments_main')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
