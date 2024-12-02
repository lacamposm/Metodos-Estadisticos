import warnings
import pandas as pd
import numpy as np
import plotly.express as px
from factor_analyzer import FactorAnalyzer

warnings.filterwarnings('ignore')


def view_loadings(fa: FactorAnalyzer, df: pd.DataFrame) -> pd.DataFrame:
    """
    Esta función toma una instancia de FactorAnalyzer y un DataFrame, y devuelve un DataFrame que muestra las cargas
    de los factores.

    :param fa: Una instancia de la clase FactorAnalyzer de la librería factor\_analyzer.
    :param df: El DataFrame de entrada utilizado para entrenar el modelo de FactorAnalyzer.
    :return: Un DataFrame que muestra las cargas de los factores. Las columnas son etiquetadas como "factor1",
    "factor2", etc., y las filas son etiquetadas con los nombres de las columnas del DataFrame de entrada.
    """
    loadings, n_factors = fa.loadings_, fa.get_params()["n_factors"]
    columns = [f"factor{i}" for i in range(1, n_factors+1)]
    df_loadings = pd.DataFrame(loadings, index=df.columns, columns=columns)
    
    return df_loadings


def get_communa_uniquess(fa: FactorAnalyzer, df_fa: pd.DataFrame) -> pd.DataFrame:
    """
    Esta función toma una instancia de FactorAnalyzer y un DataFrame de las cargas de los factores, y devuelve un
    DataFrame que muestra las comunalidades y la especificidad de cada variable en el DataFrame de entrada.

    :param fa: Una instancia de la clase FactorAnalyzer de la librería factor_analyzer.
    :param df_fa: El DataFrame de las cargas de los factores devuelto por la función view_loadings.
    :return: Un DataFrame que muestra las comunalidades y la especificidad de cada variable en el DataFrame de entrada.
    """
    communa, uniqueness, index  = fa.get_communalities(), fa.get_uniquenesses(), df_fa.columns
    temp1 = pd.DataFrame(communa, columns=["comunalidades"], index=index).T
    temp2 = pd.DataFrame(uniqueness, columns=["especificidad"], index=index).T
    
    return pd.concat([temp1,temp2]).T.sort_values(by=["comunalidades"], ascending=False)


def sedimentacion(fa):
    """ 
    Esta función genera un plot de sedimentación del objeto FA en el argumento
    Input: Objeto Fa (from factor_analyzer)
    Ouput: None
    """
    eigenvalues, length_evalue = fa.get_eigenvalues()[0], len(fa.get_eigenvalues()[0])
    varianza, length_var = fa.get_factor_variance()[2], len(fa.get_factor_variance()[0])
    m, n_factors = length_evalue - length_var, fa.get_params()["n_factors"]
    aux_lst = [0 for i in range(0, m)]  
    varianza = np.append(list(varianza), aux_lst)  
    str_componen = ["factor{}".format(i+1)  for i in range(0, len(eigenvalues))]
    
    df = pd.DataFrame(
        {
            "componente": str_componen,
            "eigenvalor": eigenvalues,
            "% acumulada": varianza*100
        }
    )
    
    (
        px.bar(
            df, x="componente", y="eigenvalor",
            hover_data={
                "componente":False,
                "% acumulada":True,
                "eigenvalor":":.2f",
                "% acumulada":":.2f"
                },
            hover_name = "componente",
            title = f"Gráfico de sedimentación con {n_factors} factores."
        )
        .add_hline(y=1.0, line_width=3, line_dash="dash", line_color="green")
        .show()
    )
