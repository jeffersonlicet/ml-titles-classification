import re
import sys
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer

nltk.download('stopwords')

spanish_snow = SnowballStemmer('spanish')
portuguese_snow = SnowballStemmer('portuguese')

tokenizer = WordPunctTokenizer()
remove_puntuaction = r'[^\w\s]'
numbers_regex = r'(\b)[0-9]+(\b)'

separators = ['-', '+', ',', '.', '(', ')', ':', '[', ']', '{', '}', '_', '/']

def preprocess_title(title, lang):
  title = title.lower()
  title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf8')

  for separator in separators:
    title = title.replace(separator, ' ')

  title = re.sub(remove_puntuaction, "", title)
  title = re.sub(numbers_regex, "", title)

  tokens = tokenizer.tokenize(title)
  stemmer = spanish_snow

  if lang == 'portuguese':
    stemmer = portuguese_snow

  return [stemmer.stem(token) for token in tokens if token not in stopwords.words(lang) and len(token) > 1]
