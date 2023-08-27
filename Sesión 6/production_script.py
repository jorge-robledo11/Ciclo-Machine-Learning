"""
    ¿Qué es un script en producción?

    • Un script de producción es un código fuente que se utiliza para implementar una aplicación o servicio en un entorno de producción, es decir, 
      en un ambiente en el que el software es utilizado por los usuarios finales. Estos scripts son típicamente escritos en lenguajes de programación 
      como Python, Java, JavaScript, Ruby, entre otros, y se encargan de llevar a cabo una serie de tareas para el correcto funcionamiento del servicio 
      o aplicación.

    • Los scripts de producción pueden incluir tareas como la configuración del entorno de ejecución, la lectura de datos de entrada y salida, 
      el procesamiento de datos, la interacción con bases de datos, el envío y recepción de información a través de redes, y la implementación de lógica 
      de negocio.

    • Los scripts de producción son esenciales para el despliegue y mantenimiento de aplicaciones en entornos de producción, ya que permiten automatizar 
      muchas de las tareas necesarias para asegurar el correcto funcionamiento de la aplicación. También son útiles para la detección y solución de 
      problemas en caso de fallas o errores en la aplicación, ya que permiten analizar y modificar el comportamiento del software en tiempo real.
"""

########################################################################################################################################################

import pandas as pd
import numpy as np
import warnings
import os
warnings.simplefilter('ignore')

# Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ingeniería de variables
from feature_engine.selection import DropFeatures
from feature_engine.imputation import AddMissingIndicator
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import RandomSampleImputer
from feature_engine.encoding import RareLabelEncoder
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OrdinalEncoder

# Selección de variables
from feature_engine.selection import DropConstantFeatures
from feature_engine.selection import DropDuplicateFeatures
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Modelado
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV

########################################################################################################################################################

# Lectura de los datos
data = pd.read_parquet('../datasets/train_preprocesado.parquet')
test = pd.read_parquet('../datasets/test_preprocesado.parquet')

# Transponemos los índices
data.index = data['id']
test.index = test['id']

# Nos deshacemos de las variables que no aportan
data.drop(['id'], inplace=True, axis=1)
test.drop(['id', 'fecha_fraude'], inplace=True, axis=1)

# Separamos los features y el target
X = data.loc[:, data.columns != 'fraude']
y = data.loc[:, data.columns == 'fraude'].squeeze()

# Splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=25)

# Función para capturar los tipos de variables
def capture_variables(data:pd.DataFrame) -> tuple:
    
    """
    Function to capture the types of Dataframe variables

    Args:
        dataframe: DataFrame
    
    Return:
        variables: tuple
    
    The order to unpack variables:
    1. continuous
    2. categoricals
    3. discretes
    4. temporaries
    """

    numericals = list(data.select_dtypes(include = [np.int64, np.int32, np.float64, np.float32]).columns)
    categoricals = list(data.select_dtypes(include = ['category', 'object', 'bool']).columns)
    temporaries = list(data.select_dtypes(include = ['datetime', 'timedelta']).columns)
    discretes = [col for col in data[numericals] if len(data[numericals][col].unique()) <= 10]
    continuous = [col for col in data[numericals] if col not in discretes]

    variables = tuple((continuous, categoricals, discretes, temporaries))
    
    # Retornamos una tupla de listas
    return variables


# Captura de variables
continuous, categoricals, discretes, temporaries = capture_variables(data=X)

# Transformamos las discretas como categóricas
X[discretes], X_train[discretes], X_val[discretes] = [subset.loc[:, discretes].astype('category') for subset in [X, X_train, X_val]]

# Variables Continuas
# Capturemos las variables con alto porcentaje de datos faltantes (más del 5%)
continuous_more_than_5perc = [var for var in continuous if X[var].isnull().mean() > 0.05]

# Capturemos las variables con menor porcentaje de datos faltantes (menos del 5%)
continuous_less_than_5perc = [var for var in continuous if X[var].isnull().sum() > 0 and X[var].isnull().mean() <= 0.05]

# Variables Categóricas
# Capturemos las variables con alto porcentaje de datos faltantes (más del 5%)
categoricals_more_than_5perc = [var for var in categoricals if X[var].isnull().mean() > 0.05]

# Capturemos las variables con menor porcentaje de datos faltantes (menos del 5%)
categoricals_less_than_5perc = [var for var in categoricals if X[var].isnull().sum() > 0 and X[var].isnull().mean() <= 0.05]

# Variables Discretas
# Capturemos las variables con alto porcentaje de datos faltantes (más del 5%)
discretes_more_than_5perc = [var for var in discretes if X[var].isnull().mean() > 0.05]

# Capturemos las variables con menor porcentaje de datos faltantes (menos del 5%)
discretes_less_than_5perc = [var for var in discretes if X[var].isnull().sum() > 0 and X[var].isnull().mean() <= 0.05]

# Variables categóricas con alta cardinalidad y baja cardinalidad
categoricals_high_cardinality = ['cod_pais']
categoricals_low_cardinality = [var for var in categoricals if var not in categoricals_high_cardinality]

# Variables discretas con alta cardinalidad y baja cardinalidad
discretes_high_cardinality = ['nropaises']
discretes_low_cardinality = [var for var in discretes if var not in discretes_high_cardinality]


pipe = Pipeline([
    # === ELIMINACIÓN ===
    # === Temporales ===
    ('tmp_to_drop', DropFeatures(features_to_drop=temporaries)),
    
    # === IMPUTACIÓN ===
    # === Continuas ===
    ('imputer_missing_indicator', AddMissingIndicator(variables=continuous_more_than_5perc)), # Indicador de ausencia
    ('imputer_mean_continuous', MeanMedianImputer(imputation_method='mean', variables=continuous_more_than_5perc)), # Imputación por la media
    ('imputer_random_continuous_less_than_5perc', RandomSampleImputer(random_state=25, variables=continuous_less_than_5perc)), # Imputación por muestra aleatoria

    # === Categóricas ===
    ('imputer_missing_categoricals_less_than_5perc', RandomSampleImputer(variables=categoricals_less_than_5perc, random_state=42)),
        
    # === ETIQUETAS RARAS ===
    # === Categóricas ===
    ('rare_label_cat_high_cardinality', RareLabelEncoder(tol=0.05, n_categories=2, 
                                                         variables=categoricals_high_cardinality)),
    ('rare_label_cat_low_cardinality', RareLabelEncoder(tol=0.05, n_categories=4,
                                                        variables=categoricals_low_cardinality)),
    # === Discretas ===
    ('rare_label_disc_high_cardinality', RareLabelEncoder(tol=0.05, n_categories=7, 
                                                          variables=discretes_high_cardinality)),
    ('rare_label_disc_low_cardinality', RareLabelEncoder(tol=0.05, n_categories=5, 
                                                         variables=discretes_low_cardinality)),
    
    # === DISCRETIZACIÓN ===
    # === Discretizador ===
    ('discretiser', EqualFrequencyDiscretiser(q=10, variables=continuous, return_object=True)),
    
    # === CODIFICACIÓN ===
    ('encoder', OrdinalEncoder(encoding_method='ordered', variables=continuous+categoricals+discretes)), # Monotonicidad
    
    # === FILTRO BÁSICO ===
    # === Cuasi-constantes ===
    ('constant', DropConstantFeatures(tol=0.998)),
    
    # === Duplicados ===
    ('duplicated', DropDuplicateFeatures()),
    
    # === Correlacionados ===
    ('correlation', SmartCorrelatedSelection(method='pearson', cv=5)),
    
    # === FILTRO ESTADÍSTICO ===
    # === Test-Chi² ===
    ('chi2', SelectKBest(chi2, k=6))
])

# 1. Ajustemos el Pipeline con los datos de entrenamiento
pipe.fit(X_train, y_train)

# 2. Hacemos una transformación: trasladando los cambios del train a los otros conjuntos de datos
X_train = pd.DataFrame(pipe.transform(X_train), columns=pipe.get_feature_names_out(), index=X_train.index)
X_val = pd.DataFrame(pipe.transform(X_val), columns=pipe.get_feature_names_out(), index=X_val.index)
test = pd.DataFrame(pipe.transform(test), columns=pipe.get_feature_names_out(), index=test.index)

# K-Fold estratificado
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=25)

# Función de coste para evaluar el rendimiento del clasificador
def cost_function(X_train, y_train):
    
    # Instanciamos nuestro clasificador
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4, class_weight='balanced')
    rf.fit(X_train, y_train)

    # Resultados de la validación cruzada
    scores = cross_validate(rf,
                            X_train,
                            y_train,
                            scoring='roc_auc',
                            cv=skfold,
                            return_estimator=True,
                            return_train_score=True,
                            n_jobs=-1)

    # Obtenemos el mejor estimador entrenado
    best_model = [estimator for idx, estimator in enumerate(scores['estimator']) if idx == np.argmax(scores['test_score'])][0]

    return best_model


# Función de coste
model = cost_function(X_train, y_train)

# Calibrar el modelo
calibrated_model = CalibratedClassifierCV(model, cv=skfold, method='isotonic', n_jobs=-1)
calibrated_model.fit(X_val, y_val)
calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]

# Predicciones con el conjunto de test
pred_final = calibrated_model.predict_proba(test)
pred_final = pd.DataFrame(pred_final, columns=[0, 1], index=test.index)
pred_final = pred_final.reset_index()

try:
    # Obtener la ruta actual del directorio de trabajo
    ruta_actual = os.getcwd()

    # Obtener el directorio padre de la ruta actual
    directorio_padre = os.path.dirname(ruta_actual)

    # Concatenar el nombre de la carpeta "datasets" al directorio padre
    ruta_datasets = os.path.join(directorio_padre, 'datasets')

    # Concatenar el nombre de la carpeta "resultados" a la ruta "ruta_datasets"
    path = os.path.join(ruta_datasets, 'resultados')

    # Crear la carpeta "resultados" en la ruta "path". El argumento exist_ok=True le indica a Python que no genere un error si la carpeta ya existe
    os.makedirs(path, exist_ok=True)

    # Exportar los resultados como un "csv"
    pred_final.to_csv(f'{path}\predicciones.csv', index=False)
    
except Exception as e:
        print(type(e).__name__)
        
finally:
    print('¡Realizado exitosamente!')