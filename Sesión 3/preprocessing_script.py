import pandas as pd
import numpy as np
import re


for set in ['train', 'test']:
    
    try:
        # Cargar y lectura de los datos
        path = f'../datasets/{set}.csv'
        data:pd.DataFrame = pd.read_csv(path)
        print('1. Carga y lectura realizada exitosamente')

        # Manejar el nombre de los predictores
        data = data.rename(columns=lambda col: str(col).lower().strip())
        print('2. Renombramiento de los predictores realizado exitosamente')

        # Eliminar duplicados
        data.drop_duplicates(inplace=True, ignore_index=True)
        print('3. Remoción de duplicados realizado exitosamente')

        # Reemplazar valores faltantes de distintas fuentes a np.nan
        data = data.fillna(np.nan)
        data = data.replace({'ERROR': np.nan, '': np.nan, 'None': np.nan, 'n/a': np.nan,
                             'N/A': np.nan, 'NULL': np.nan, 'NA': np.nan, 'NAN': np.nan})
        print('4. Reemplazo de valores faltantes de distintas fuentes realizado exitosamente')

        # Transformar los predictores temporales y cambiar su formato
        t = data.columns[data.columns.str.contains('fecha')]
        data[t] = data[t].apply(lambda x: pd.to_datetime(x)).apply(lambda x: x.dt.strftime('%Y-%m-%d')).apply(lambda x: pd.to_datetime(x))
        print('5. Transformación de predictores temporales realizado exitosamente')

        # Uniformizar los predictores categóricos
        categoricals = list(data.select_dtypes(include = ['object', 'bool']).columns)
        data[categoricals] = data[categoricals].fillna(np.nan).applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        data[categoricals] = data[categoricals].astype('category')
        print('6. Transformación de predictores categóricos realizado exitosamente')

        # Remover predictores que su distribución supera el 33.33% como datos faltantes
        data.drop(data.columns[data.isnull().mean() > 1/3].to_list(), inplace=True, axis=1)
        print('7. Remoción de predictores con valores faltantes en su distribución superior a 1/3 realizado exitosamente')

        # Castear los predictores identificadores
        regex = r'id'
        id = [col for col in data.columns if re.search(regex, col)]
        data[id] = data[id].astype('category')
        print('8. Transformación de predictores identificadores realizado exitosamente')

        # Exportar los datos preprocesados
        data.to_parquet(f'../datasets/{set}_preprocesado.parquet', index=False)
        print('9. Exportación de los datos pre-procesados realizado exitosamente')
        
    except Exception as e:
        print(type(e).__name__)
        
    finally:
        print('\n¡Pre-procesamiento realizado exitosamente!')
