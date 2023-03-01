import asyncio
import re
from tqdm import tqdm
import requests
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

# https://steamcommunity.com/app/570/reviews/
main_page = 'https://store.steampowered.com/app/570/Dota_2/'
path_main = 'game_app/static/game_app/reviews/'

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/108.0.0.0 Safari/537.36', 'Accept-language': 'ru-RU,ru;q=0.9,be;q=0.8,el;q=0.7,'
                                                                              'en;q=0.6', 'Accept': r'*/*'}


def get_reviews(url, params):
    r = requests.get(url=f'https://store.steampowered.com/appreviews/{url[35:]}', headers=headers,
                     params=params).json()
    return r


def get_n_reviews(url, n=100):
    reviews = []
    cursor = '*'
    params = {'json': 1,
              'filter': 'all',
              'language': 'english',
              'review_type': 'all',
              'purchase_type': 'all',
              'num_per_page': 100}
    while n > 0:
        params['cursor'] = cursor.encode()
        n -= 100
        response = get_reviews(url, params)
        cursor = response['cursor']
        for review in response['reviews']:
            reviews.append(review['review'])
        if len(response['reviews']) < 100:
            break

    return reviews


def clear_text(text: str) -> str:
    """
  Функция получает на вход строчку текста

  Удаляет с помощью регулярного выражения
  все не кириллические символы и приводит
  слова к нижнему регистру

  Возвращает обработанную строчку
  """
    # Пишем регулярное выражение которое заменяет на ' '
    # все, что не входит в кириллический алфавит
    clear_text = re.sub(r'[^A-z]+', ' ', text).lower()
    return ' '.join(clear_text.split())


def clean_stop_words(text: str, stopwords: list):
    """
  Функция получает:
  * text -- строчку текста
  * stopwords -- список стоп слов для исключения
  из текста

  Возвращает строчку текста с исключенными стоп словами

  """
    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)


def lemmatize(df: (pd.Series, pd.DataFrame), text_column: (None, str), n_samples: int, break_str='br', ) -> pd.Series:
    """
    Принимает:
    df -- таблицу или столбец pandas содержащий тексты,
    text_column -- название столбца указываем если передаем таблицу,
    n_samples -- количество текстов для объединения,
    break_str -- символ разделения, нужен для ускорения,
    количество текстов записанное в n_samples объединяется
    в одит большой текст с предварительной вставкой символа
    записанного в break_str между фрагментами
    затем большой текст лемматизируется, после чего разбивается на
    фрагменты по символу break_str


    Возвращает:
    Столбец pd.Series с лемматизированными текстами
    в которых все слова приведены к изначальной форме:
    * для существительных — именительный падеж, единственное число;
    * для прилагательных — именительный падеж, единственное число,
    мужской род;
    * для глаголов, причастий, деепричастий — глагол в инфинитиве
    (неопределённой форме) несовершенного вида.

    """

    result = []
    m = Mystem()
    if df.shape[0] % n_samples == 0:
        n_iterations = df.shape[0] // n_samples
    else:
        n_iterations = (df.shape[0] // n_samples) + 1
    for i in tqdm(range(n_iterations)):
        start = i * n_samples
        stop = start + n_samples
        sample = break_str.join(df[text_column][start: stop].values)
        lemmas = m.lemmatize(sample)
        lemm_sample = ''.join(lemmas).split(break_str)
        result += lemm_sample

    return pd.Series(result, index=df.index)


def analyze_comment(game_urls, n=100):
    def get_proba(url):
        reviews_df = pd.DataFrame({"comment": get_n_reviews(url, n)})
        reviews_df['text_clear'] = reviews_df['comment'].apply(
            lambda x: clean_stop_words(clear_text(str(x)), stopwords))
        reviews_tf_idf = count_idf_1.transform(reviews_df['text_clear'])
        reviews_neg_proba = model_lr_base_1.predict_proba(reviews_tf_idf)
        reviews_df['negative_proba'] = reviews_neg_proba[:, 0]
        reviews_neg_percent = (reviews_df['negative_proba'] > 0.50).sum() / reviews_df.shape[0]
        return reviews_neg_percent

    stopwords = set(nltk_stopwords.words('english'))
    comments = pd.read_csv(path_main + '\\comments_clean.csv', sep=',', header=None)
    comments = pd.DataFrame(comments.iloc[:, 1:])
    comments.columns = ['text', 'label']
    comments.index = range(comments.shape[0])
    train, test = train_test_split(comments, test_size=0.2, random_state=5468, )
    count_idf_1 = TfidfVectorizer(ngram_range=(1, 1))
    ctf_idf_base_1 = count_idf_1.fit(comments['text'])
    tf_idf_train_base_1 = count_idf_1.transform(train['text'])
    tf_idf_test_base_1 = count_idf_1.transform(test['text'])
    model_lr_base_1 = LogisticRegression(solver='lbfgs', random_state=12345, max_iter=10000, n_jobs=-1)
    model_lr_base_1.fit(tf_idf_train_base_1, train['label'])
    predict_lr_base_proba = model_lr_base_1.predict_proba(tf_idf_test_base_1)
    matrix = confusion_matrix(test['label'], (predict_lr_base_proba[:, 0] < 0.454545).astype('int'), normalize='true')

    return [get_proba(url) for url in game_urls]
