import re
import os
import nltk
import pandas as pd

from loguru import logger
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')

class text_preprocessor():

    def __init__(self):
        self.path_to_data = os.path.join(os.getcwd(), '../data/payments_training.tsv')
        self.columns = ['id', 'date', 'payment', 'text', 'class']
        self.drop_columns = ['id', 'date', 'class']
        PCA_components = 199  # !
        self.PCA_transform = PCA(n_components=PCA_components)
        self.stopwords = stopwords.words("russian")
        self.stopwords.extend(
            {"по", "от", "до", "с", "в", "на", "за", "к", "о", "об", "у", "со", "из", "при", "под", "про", "через",
             "над", "без"})
        self.morph = MorphAnalyzer()

    def __load_csv(self):
        file_path = os.path.join(os.getcwd(), '../data/payments_training.tsv')
        self.df = pd.read_csv(file_path, index_col=False, sep='\t', names=self.columns)
        self.df = self.df.drop(columns=self.drop_columns)

    def remove_non_cyrillic(self, text):
        cyrillic_pattern = r"[^а-яА-ЯёЁ]+"
        cleaned_text = re.sub(cyrillic_pattern, " ", text)
        return cleaned_text

    def split_joined_words(self, text):

        tokens = []

        def split_token(split_points, text):
            for i in range(1, len(split_points)):
                token = text[split_points[i - 1]:split_points[i]]
                tokens.append(f'{token} ')

        splited_text = text.split(' ')
        for token in splited_text:
            split_points = [0]
            for i in range(1, len(token)):
                token = token.strip()
                if not token[i - 1].isupper() and token[i].isupper():
                    split_points.append(i)

            split_points.append(len(token))
            split_token(split_points, token)

        return ''.join(tokens)

    def remove_stop_words(self, text):
        text_splited = text.split(' ')
        filtered_tokens = []
        for token in text_splited:
            if token not in self.stopwords:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]
                if len(token) < 2:  # Отсев единичных букв
                    continue
                filtered_tokens.append(f'{token} ')

        return ''.join(filtered_tokens)

    def convert_to_tfdif(self, texts):
        texts_raw = texts.values
        unique_tokens = []
        for text in texts_raw:
            tokens = text.split(' ')
            for token in tokens:
                if token not in unique_tokens:
                    unique_tokens.append(token)

        count_unique_tokens = len(unique_tokens)
        tfidf = TfidfVectorizer(max_features=count_unique_tokens)
        mat = tfidf.fit_transform(texts).toarray()
        scaler = StandardScaler()
        return scaler.fit_transform(mat)
        # return mat

    def apply_PCA(self, tfidf_matrix):
        tfidf_matrix_compressed = self.PCA_transform.fit_transform(tfidf_matrix)
        return tfidf_matrix_compressed

    def get_data(self):
        self.__load_csv()
        self.df['text'] = self.df['text'].apply(self.remove_non_cyrillic)
        self.df['text'] = self.df['text'].apply(self.split_joined_words)
        self.df['text'] = self.df['text'].apply(lambda x: x.lower())
        self.df['text'] = self.df['text'].apply(self.remove_stop_words)
        tfidf_mat = self.convert_to_tfdif(self.df['text'])
        tfidf_mat_compr = self.apply_PCA(tfidf_mat)
        return tfidf_mat_compr

    def run(self):
        processed_data = self.get_data()
        logger.info(f"Processed data shape: {processed_data.shape}")
        if len(processed_data.shape) != 2:
            raise ValueError("Processed data is not in the expected format!")
        processed_df = pd.DataFrame(processed_data)
        output_path = os.path.join(os.getcwd(), '../data', 'processed_data.tsv')
        processed_df.to_csv(output_path, sep='\t', index=False)
        logger.info("Successfully saved")


if __name__ == '__main__':
    try:
        logger.info('START_PREPROCESSING')
        data_processor = text_preprocessor()
        data_processor.run()
        logger.info('END_PREPROCESSING')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

