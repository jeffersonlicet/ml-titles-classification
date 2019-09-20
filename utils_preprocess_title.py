import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from string import digits
import unicodedata

nltk.download('stopwords')

spanishSnow = SnowballStemmer('spanish')
portugueseSnow = SnowballStemmer('portuguese')

tokenizer = WordPunctTokenizer()
regex = r'[^\w\s]'
nonumber = r'\w*\d\w*'

_with = r'(\b)c\/'
_without = r'(\b)s\/'
_com = r'(\b)com(\b)'
_sem = r'(\b)sem(\b)'
numb = r'(\b)[0-9]*(\b)'

separators = [
  '-',
  '+',
  ',',
  '.',
  '(',
  ')',
  ':',
  '[',
  ']',
  '{',
  '}',
]

def preprocess_title(title, lang):
  title = title.lower()
  title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf8')
  for separator in separators:
    title = title.replace(separator, ' ')

  title = re.sub(_with, "con ", title)
  title = re.sub(_without, "sin ", title)
  title = re.sub(_com, "con ", title)
  title = re.sub(_sem, "sin ", title)
  title = re.sub(regex, " ", title)
  title = re.sub(numb, " ", title)

  tokens = tokenizer.tokenize(title)
  stemmer = spanishSnow
  if lang == 'portuguese':
    stemmer = portugueseSnow

  return [stemmer.stem(token) for token in tokens if len(token) > 1]
