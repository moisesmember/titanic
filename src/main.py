import pandas as pd
import warnings
from src.clean_data.index import clean
from src.data_preprocessing.index import preprocessing, correlated_columns
from src.explore_data.index import explore

# Ignorar o aviso específico
warnings.filterwarnings("ignore", message="Setting an item of incompatible dtype")

# Read data
titanic_origin = pd.read_excel("./database/titanic3.xls")

# Clean data
new_titanic = clean(titanic_origin)

# Explore data
explore(new_titanic)

preprocessing(new_titanic)

# Restaurar o comportamento padrão de mensagens de aviso
warnings.resetwarnings()