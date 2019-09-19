import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from string import digits
import unicodedata

nltk.download('stopwords')

colors = [
  'si',
  'no',
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
nonumber = r'\w*\d\w*'

_3D = r'[0-9]+x[0-9]+x[0-9]+'
_2D = r'[0-9]+x[0-9]+'
_mm = r'[0-9]+x[0-9]+'
_m = r'[0-9]+mm'
_grs = r'[0-9]+(gr)s*'
_grs_alone = r'(\b)grs*(\b)'
_with = r'(\b)c\/'
_without = r'(\b)s\/'
remove_digits = str.maketrans('', '', digits)

separators = [
  '-',
 # 'c/',
 # 's/',
 # '/',
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
_com = r'(\b)com(\b)'
_sem = r'(\b)sem(\b)'
def preprocess_title(title, lang):
  #print(title)
  title = title.lower()
  title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf8')
  #title = title.replace('-', ' ').replace('c/', ' ').replace('s/', ' ').replace('/', ' ').replace('+', ' ').replace(',', ' ').replace('.', ' ').replace('(', ' ').replace(')', ' ')
  for separator in separators:
    title = title.replace(separator, ' ')

  title = re.sub(_with, "con ", title)
  title = re.sub(_without, "sin ", title)
  title = re.sub(_com, "con ", title)
  title = re.sub(_sem, "sin ", title)
  title = re.sub(regex, " ", title)

 #title = re.sub(_2D, "medida", title)
  #title = re.sub(_m, "medida", title)
  #title = re.sub(_grs, "gramos", title)
  #title = re.sub(_grs_alone, "gramos", title)
  #title = re.sub(nonumber, "", title)
  #tokens = []
  tokens = tokenizer.tokenize(title)
  stemmer = spanishSnow
  if lang == 'portuguese':
    stemmer = portugueseSnow

  return [stemmer.stem(token) for token in tokens]

  """
  token = token.strip()

  # Comment here to generate large
  title = re.sub(_3D, "medida", title)
  title = re.sub(_2D, "medida", title)
  title = re.sub(_m, "medida", title)
  title = re.sub(_grs, "gramos", title)
  title = re.sub(_grs_alone, "gramos", title)
  title = re.sub(nonumber, "", title)
  # Comment to here to generate large
  tokens = []
  for token in tokenizer.tokenize(title):
    token = token.strip()

    if token not in stopwords.words(lang) and len(token) >= 2 and token not in colors:
      tokens.append(token)

  stemmer = spanishSnow
  if lang == 'portuguese':
    stemmer = portugueseSnow

  titleStemmed = [stemmer.stem(token) for token in tokens]
  return titleStemmed
  """
