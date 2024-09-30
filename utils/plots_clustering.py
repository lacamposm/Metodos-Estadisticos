import numpy as np
import pandas as pd
import plotly.express as px
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def cluster_biplot(pca, dataframe, col_clusters, comp1=1, comp2=2, size_text=8):
    """
    Esta función construye el biplot de un PCA y muestra los clusters en el plano factorial.

    :param pca: Objeto PCA que se usará para plotear los clusters.
    :param dataframe: pandas.DataFrame con el que se realizó el PCA.
    :param col_cluster: Lista o pd.Series con las etiquetas asignadas vía el método de cluster.
    :param comp1: Componente en el eje x.
    :param comp2: Componente en el eje y.
    :param size_text: int, opcional. Tamaño del texto para evitar overlapping (default: 10).
    :return: None
    """
    comp_1, comp_2 = str(comp1), str(comp2)  ## Strings de las compomentes a plotear.
    scaler, length = StandardScaler(), len(pca.explained_variance_)
    percent_var = pca.explained_variance_ratio_ * 100
    scaler.fit(dataframe)
    X_scaled = scaler.transform(dataframe)
    ##
    tmp = dataframe.copy()
    if tmp.index.name == None:  ## En caso que el DataFrame no tenga nombre para el índice.
        tmp.index.name = "Indice"
    pca_trans = pd.DataFrame(pca.transform(X_scaled), index=tmp.index,
                             columns=["PC" + str(comp) for comp in range(1, length + 1)])
    text_list = [pca_trans.index.name + ": {}".format(pca_trans.index[i]) for i in range(0, len(tmp))]

    features, tmp["cluster"] = tmp.columns, col_clusters
    tmp["cluster"] = tmp["cluster"].astype("category")
    ##
    fig = px.scatter(tmp, x=pca_trans[f"PC{comp_1}"], y=pca_trans[f"PC{comp_2}"], color="cluster",
                     text=tmp.index, hover_name=tmp.index, template="plotly_white", symbol="cluster")
    ## Personalización plot
    fig.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.update_traces(textposition=improve_text_position(pca_trans[f"PC{comp_1}"]),
                      textfont_size=size_text, )
    fig.update_layout(title="PCA-CLUSTER Biplot.")
    fig.update_xaxes(range=[min(pca_trans[f"PC{comp_1}"] - 0.35), max(pca_trans[f"PC{comp_1}"]) + 0.35],
                     title_text="Dim " + comp_1 + " ({:.2f}%)".format(percent_var[comp1 - 1]))
    fig.update_yaxes(range=[min(pca_trans[f"PC{comp_2}"] - 0.35), max(pca_trans[f"PC{comp_2}"]) + 0.35],
                     title_text="Dim " + comp_2 + " ({:.2f}%)".format(percent_var[comp2 - 1]))
    fig.show()

    return None


def improve_text_position(x):
    """
    Esta función intercala las etiquetas en el texto de un plot en plotly
    """
    positions = ["top center", "bottom center"]
    return [positions[i % len(positions)] for i in range(len(x))]


def cophenetic_coef(df, methods, metrics):
    """
    Función que permite determinar el coeficiente de aglomeración entre varias métricas y métodos.

    :param df: pd.DataFrame ya estandarizada y con solo variables cuantitativas.
    :param methods: list. Contiene la lista de métodos que se usará para comparar.
    :param metrics: list. Contiene la lista de las métricas que se usará para comparar.
    :return: pd.DataFrame que contendrá el cálculo de cada coeficiente de aglomeración,
             donde los métodos son las columnas y las métricas los índices. Dado el caso que
             el resultado sea -1 es que métrica y método genera error.
    """
    result = pd.DataFrame(None, columns=methods, index=metrics)
    for metric in metrics:
        original = pdist(df, metric=metric)
        for method in methods:
            try:
                tmp = sch.linkage(df, method=method, metric=metric)
                cophenet = sch.cophenet(tmp)
                result.loc[metric, method] = np.corrcoef(cophenet, original)[0, 1]
            except:
                result.loc[metric, method] = -1
    return result


def optimal_clusters_hierarchy(df, metrics, to_sklearn=False):
    """
    Función que determina el número óptimo de clusters, el mejor método de linkage y la mejor métrica
    basándose en el coeficiente de correlación cofenética y el corte óptimo en el dendrograma.
    
    :param df: pd.DataFrame ya estandarizada y con solo variables cuantitativas.
    :param metrics: list. Lista de métricas de distancia a comparar.
    
    :return: tuple con (n_clusters, mejor_métrica, mejor_linkage, umbral_optimo)
    """
    methods = ["ward", "complete", "average", "single"]
      
    result = cophenetic_coef(df, methods, metrics)
    best_metric = result.max(axis=1).idxmax()
    best_method = result.max().idxmax()
        
    best_linkage = sch.linkage(df, method=best_method, metric=best_metric)    
    df_temp = pd.DataFrame(best_linkage[:, 2], columns=["heights"])
    df_temp["diff"] = df_temp.diff(1)
    df_temp = df_temp.sort_values(by="diff", ascending=False)
    optimal_threshold = df_temp["heights"].head(2).mean()
    clusters = sch.fcluster(best_linkage, t=optimal_threshold, criterion='distance')
    if to_sklearn:
        return {
        "n_clusters": len(np.unique(clusters)),
        "metric": best_metric,
        "linkage": best_method,
    }
    
    else:
        {
        "n_clusters": len(np.unique(clusters)),
        "metric": best_metric,
        "linkage": best_method,
        "optimal_threshold": optimal_threshold,
        "cophenetic": result.max().max()
    }
