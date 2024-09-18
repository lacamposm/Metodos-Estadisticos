import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (15, 9)
plt.style.use("ggplot")


def biplot(pca=None, dataframe=None, comp1: int = 1, comp2: int = 2):
    """
    Esta función construye el biplot del correspodiente pca.

    Input:  pca: Objeto pca que se desea plotear.
            dataframe: pandas-DataFrame con el que se realizó el pca.
            comp1 = Componente en el eje x.
            comp2 = Compomente en el eje y
    """

    comp_1, comp_2 = str(comp1), str(comp2)
    vectores = pca.components_[[comp1 - 1, comp2 - 1]].T * np.sqrt(pca.explained_variance_[[comp1 - 1, comp2 - 1]])
    scaler, length = StandardScaler(), len(pca.explained_variance_)
    scaler.fit(dataframe)
    X_scaled = scaler.transform(dataframe)
    if dataframe.index.name is None:  # En caso que el DataFrame no tenga nombre para el índice.
        dataframe.index.name = "Indice"
    pca_trans = pd.DataFrame(pca.transform(X_scaled), index=dataframe.index,
                             columns=["PC" + str(comp) for comp in range(1, length + 1)])
    text_list = [pca_trans.index.name + ": {}".format(pca_trans.index[i]) for i in range(0, len(dataframe))]

    features = dataframe.columns

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pca_trans["PC" + comp_1], y=pca_trans["PC" + comp_2], mode="markers",
                             hovertemplate="<b>%{text}</b>" + "<br><br><b>PCX</b>: %{x:.2f}<br>" +
                                           "<b>PCY</b>: %{y:.2f}<br>", name="", text=text_list, ))
    # Construcción del biplot.
    for count, feature in enumerate(features):
        fig.add_annotation(x=vectores[count, 0], y=vectores[count, 1], ax=0, ay=0,
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor="black")

        fig.add_annotation(x=vectores[count, 0], y=vectores[count, 1], ax=0, ay=0, xanchor="center",
                           text=feature, showarrow=True, arrowhead=3, arrowcolor="rgb(255,51,0)",
                           font=dict(family="Courier New, monospace", size=10, color="black"),
                           yanchor="bottom", )

    # Personalización plot
    percent_var = pca.explained_variance_ratio_ * 100
    fig.add_hline(y=0, line_color="black")
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(title="Biplot.")
    fig.update_xaxes(range=[min(pca_trans["PC" + comp_1] - 0.2), max(pca_trans["PC" + comp_1]) + 0.2],
                     title_text="Dim " + comp_1 + " ({:.2f}%)".format(percent_var[comp1 - 1]))
    fig.update_yaxes(range=[min(pca_trans["PC" + comp_2] - 0.2), max(pca_trans["PC" + comp_2]) + 0.2],
                     title_text="Dim " + comp_2 + " ({:.2f}%)".format(percent_var[comp2 - 1]))
    fig.show()


def biplot1(pca=None, dataframe: pd.DataFrame = None):
    """
    Esta función construye el biplot del correspodiente pca solo mostranto las proyecciones de las variables originales
    Input:  pca: Objeto que pca que se desea plotear.
            dataframe: pandas-DataFrame con el que se realizo el pca.
    Output: None.
    """
    fig = go.Figure()
    # Circulo unitario.
    fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line_color="blue", line_width=0.5,
                  line_dash="solid", )
    # Trabajo con el pca.
    vectores = pca.components_.T * np.sqrt(pca.explained_variance_)
    features = dataframe.columns
    # construcción del plot.
    for count, features in enumerate(features):
        fig.add_annotation(x=vectores[count, 0], y=vectores[count, 1], ax=0, ay=0,
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor="black")

        fig.add_annotation(x=vectores[count, 0], y=vectores[count, 1], ax=0, ay=0, xanchor="center",
                           text=features, showarrow=True, arrowhead=3, arrowcolor="rgb(255,51,0)",
                           font=dict(family="Courier New, monospace", size=10, color="black"),
                           yanchor="bottom", )
    # Personalización plot
    percent_var = pca.explained_variance_ratio_ * 100
    fig.update_layout(width=500, height=500, title="Primer plano factorial.")
    fig.update_xaxes(range=[-1, 1], title_text="Dim 1" + " ({:.2f}%)".format(percent_var[0]))
    fig.update_yaxes(range=[-1, 1], title_text="Dim 2" + " ({:.2f}%)".format(percent_var[1]))
    fig.show()


def sedimentacion(pca):
    """
    Esta función genera un plot de sediemntación del objeto pca en el argumento

    Input: Objeto pca (from sklearn PCA)
    """
    expl_var = pca.explained_variance_ratio_
    str_componen = ["PC{}".format(i + 1) for i in range(0, len(expl_var))]
    df = pd.DataFrame({"componente": str_componen,
                       "% varianza": expl_var * 100,
                       "% acumulada": np.cumsum(expl_var * 100)})
    fig = px.bar(df, x="componente", y="% varianza",
                 hover_data={"componente": False,
                             "% acumulada": True,
                             "% varianza": ":.2f",
                             "% acumulada": ":.2f"},
                 hover_name="componente",
                 title="Gráfico de sedimentación"
                 )
    fig.show()


def view_index(df: pd.DataFrame, indexs):
    """
    Esta función muestra la información del dataframe df proporcionando una lista de indexs.

    Input: df: pd.DataFrame
    indexs: Lista con el índice o índices a seleccionar del DataFrame.
    Output: pd.DataFrame con la información del índice o índices.
    """
    if len(indexs) == 1:
        return df.loc[indexs[0]].to_frame().T
    else:
        return pd.concat([df.loc[index,] for index in indexs], axis=1).T


def plot_loadings_heatmap(pca, df_pca, n_compo: str = 2):
    """
    Esta función genera un heatmap de los loadings de las componentes pricipales que se tienen
    en el objeto pca que se recibe.

    Input:
    pca: Objeto PCA (sklearn)
    df_pca: pandas DataFrame con quien se realizo el pca. O una lista de los nombres de igual cantidad que las
            compomentes principales del objeto pca ingresado.
    n_compo: Número de componentes principales, a las cuales se les desea ver sus loadings.
    """
    components = pca.components_[:n_compo]
    if isinstance(df_pca, pd.DataFrame):
        columns = df_pca.columns
    else:
        columns = df_pca

    yticks = [f"ACP{i}" for i in range(1, n_compo + 1)]
    df = pd.DataFrame(components, columns=columns, index=yticks)

    px.imshow(
        df,
        labels=dict(x="Features", y="Principal Components", color="Loadings"),
        title="Heatmap loadings",
        text_auto=True,
    ).show()


def plot_loadings_bar(pca, df_pca: pd.DataFrame, n_compo: int = 2, rotation: int = 90):
    """
    Esta función genera un heatmap de los loadings de las componentes pricipales que se tienen
    en el objeto pca que se recibe.

    Input:
    pca: Objeto PCA (sklearn)
    df_pca: pandas DataFrame con quien se realizo el pca. O una lista de los nombres
            de igual cantidad que las compomentes principales del objeto pca ingresado.
    n_compo: Número de componentes principales, a las cuales se les desea ver sus loadings.
    rotation: Grado de rotación de los xticks.

    Output: None
    """
    components = pca.components_[:n_compo]
    loadings = pd.DataFrame(components, columns=df_pca.columns)
    loadings.index = [f"PC{i + 1}" for i in range(0, n_compo)]
    loadings = loadings.T
    max_eje_y = np.max(loadings.abs().max())

    if n_compo % 2 == 0:
        rows = int(n_compo / 2)
    else:
        rows = int(n_compo / 2) + 1

    fig = make_subplots(rows=rows, cols=2, shared_yaxes=True, subplot_titles=[f"PC{i + 1}" for i in range(n_compo)])

    for i in range(n_compo):
        row = i // 2 + 1
        col = i % 2 + 1
        temp = loadings[f"PC{i + 1}"].to_frame()
        temp["color"] = np.where(temp[f"PC{i + 1}"] < 0, "red", "blue")
        fig.add_trace(go.Bar(x=temp.index, y=temp[f"PC{i + 1}"], marker_color=temp["color"]), row=row, col=col)
        fig.add_shape(type="line", x0=-0.5, x1=len(temp) - 0.5, y0=0, y1=0, line=dict(color="gray"), row=row, col=col)
        fig.update_xaxes(tickangle=rotation, row=row, col=col)

    fig.update_layout(title_text=f"Loadings para las primeras {n_compo} componentes principales.",
                      height=rows * 370, width=1300, showlegend=False)
    fig.update_yaxes(range=[-1.01 * max_eje_y, 1.01 * max_eje_y])
    fig.show()
