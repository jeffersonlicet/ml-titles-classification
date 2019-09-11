import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
nltk.download('stopwords')

colors = [
  'rojo',
  'naranja',
  'amarillo',
  'verde',
  'azul',
  'púrpura',
  'morado',
  'marrón',
  'negro',
  'gris',
  'blanco',
  'índigo',
  'rosa',
  'laranja',
  'amarelo',
  'vermelho',
  'preto',
  'marrom',
  'castanho',
  'roxo',
  'branco',
  'cinza',
  'dourado',
  'prateado',
]

spanishSnow = SnowballStemmer('spanish')
portugueseSnow = SnowballStemmer('portuguese')

tokenizer = WordPunctTokenizer()
regex = r'[^\w\s]'

def preprocess_title(title, lang):
  title = title.lower()
  title = title.replace('-', ' ')
  title = title.replace('/', ' ')
  title = re.sub(regex, "", title)
  title = re.sub(r'(\s|\b)[0-9]+\b', "", title)
  tokens = [token for token in tokenizer.tokenize(title) if token not in stopwords.words(lang) and len(token) >= 3 and token not in colors]

  stemmer = spanishSnow
  if lang == 'portuguese':
    stemmer = portugueseSnow

  title = [stemmer.stem(token) for token in tokens]
  #title = tokens
  return title
