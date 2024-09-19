import prince
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def improve_text_position(x):
    """"
    Esta función intercala las etiquetas en el texto de un plot en plotly
    """
    positions = ["top center", "bottom center"]
    return [positions[i % len(positions)] for i in range(len(x))]


def get_eigenvalue(ca: prince.CA) -> pd.DataFrame:
    """
    Esta función muestra la inercia para cada una de las componetes.
    :param ca: Objecto CA de la libreria prince.
    :return DataFrame con los eigenvalues, varianza y varianza acumulada de cada dimensión.
    """

    eigen, var, = np.around(ca.eigenvalues_, decimals=3), np.array(ca.percentage_of_variance_)
    n_compo, var = ca.n_components, np.around(var, decimals=3)
    indexs = [f"Dim{i + 1}" for i in range(0, n_compo)]

    return pd.DataFrame(
        {
            "eigenvalue": eigen, "variance_percent": var,
            "cumulative_variance_percent": np.cumsum(var)
        },
        index=indexs
    )


def get_cos2(ca: prince.CA, df_ca: pd.crosstab, choice: str = "index"):
    """
    Función que encuentran la cantidad de representación de las filas (columnas) en el 
    correspondiente factor.

    :param ca: Objeto CA de la librería prince.
    :param df_ca: pd.crosstab con quien se ajustó el CA del parámetro ca. DEBE ser la tabla de contingencias.
    :param choice: Eleccion de los cos2. Puede ser "index" o "columns" Default = "index"
    :return pd.DataFrame con los cos2.
    """
    if choice == "index":
        cos2 = ca.row_cosine_similarities(df_ca)
        return cos2.rename(columns={i: f"Dim{i + 1}" for i in cos2.columns})

    elif choice == "columns":
        cos2 = ca.column_cosine_similarities(df_ca)
        return cos2.rename(columns={i: f"Dim{i + 1}" for i in cos2.columns})


def screeplot(ca: prince.CA):
    """
    Esta función genera el gráfico de sedimentación del análisis de correspondencias.
    :param ca: Objeto CA de la libreria prince.
    """
    inertia = ca.percentage_of_variance_

    df = pd.DataFrame(
        {
            "Dimension": [str(i + 1) for i in range(len(inertia))],
            "Percentage of explained variance": np.array(inertia)
        }
    )

    (
        px.bar(df, x="Dimension", y="Percentage of explained variance", color_discrete_sequence=["blue"],
               title="Scree plot")
        .add_hline(y=np.mean(inertia), line_dash="dash", line_color="red")
        .add_trace(go.Scatter(
            x=df["Dimension"],
            y=df["Percentage of explained variance"],
            marker=dict(color="black", symbol="circle"),
            showlegend=False
        )
        )
        .show()
    )


def ca_plot_cos2(ca: prince.CA, df_ca: pd.DataFrame, choice: str = "index"):
    """
    Esta función plotea el cos2 de un análisis de correspondencias (CA).
    
    :param ca: Objeto CA de la libreria prince.
    :param df_ca: pd.DataFrame con quien se realizó el ajuste del objeto CA de la libreria prince.
    :param choice:  Selección de filas ("index") o columnas ("columns) a plotear. Default "index".
    """
    var_exp = np.round(np.array(ca.percentage_of_variance_), 2)
    if choice == "index":
        df_to_plot = ca.row_coordinates(df_ca)

    elif choice == "columns":
        df_to_plot = ca.column_coordinates(df_ca)

    df_to_plot.columns = [f"Dim{i + 1}" for i in range(len(df_to_plot.columns))]
    df_to_plot["cos2"] = get_cos2(ca, df_ca, choice=choice)[["Dim1", "Dim2"]].sum(axis=1).round(4)

    fig = px.scatter(
        data_frame=df_to_plot,
        x="Dim1",
        y="Dim2",
        text=df_to_plot.index,
        color="cos2",
        color_continuous_scale="temps",
        template="plotly_white"
    )

    fig.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.update_traces(textposition=improve_text_position(df_to_plot["Dim1"]))
    fig.update_xaxes(range=[min(df_to_plot["Dim1"]) - 0.2, max(df_to_plot["Dim1"]) + 0.2],
                     title_text=f"Dim1 ({var_exp[0]}%)")
    fig.update_yaxes(range=[min(df_to_plot["Dim2"]) - 0.2, max(df_to_plot["Dim2"]) + 0.2],
                     title_text=f"Dim2 ({var_exp[1]}%)")

    if choice == "index":
        fig.update_layout(title="Row points - CA")

    elif choice == "columns":
        fig.update_layout(title="Column points - CA")

    fig.show()


def ca_biplot(ca: prince.CA, df_ca: pd.crosstab):
    """
    Esta función genera el biplot de un objeto CA de la librería prince.
    
    :param ca: Objeto CA de la libreria prince.
    :param df_ca: pd.DataFrame con quien se realizó el ajuste del objeto CA de la libreria prince.
    """
    row, col = ca.row_coordinates(df_ca), ca.column_coordinates(df_ca)
    var_exp = np.round(np.array(ca.percentage_of_variance_), 2)
    columns_names = [f"Dim{i + 1}" for i in range(0, ca.row_coordinates(df_ca).shape[1])]
    row.columns, col.columns = columns_names, columns_names
    row["Variable"], col["Variable"] = df_ca.index.name, df_ca.columns.name
    data_plot = pd.concat([row, col])
    minx, miny = min(data_plot[f"Dim1"]) - 0.2, min(data_plot[f"Dim2"]) - 0.2,
    maxx, maxy = max(data_plot[f"Dim1"]) + 0.2, max(data_plot[f"Dim2"]) + 0.2

    (
        px.scatter(
            data_plot, x=f"Dim1", y=f"Dim2", color="Variable", symbol="Variable",
            text=data_plot.index, template="plotly_white", title="CA-Biplot",
            hover_name=data_plot["Variable"], hover_data={"Variable": False}
        )
        .add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
        .add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
        .update_traces(textposition=improve_text_position(data_plot[f"Dim1"]))
        .update_xaxes(range=[minx, maxx], title_text=f"Dim1 ({var_exp[0]}%)")
        .update_yaxes(range=[miny, maxy], title_text=f"Dim2 ({var_exp[1]}%)")
        .show()
    )


def plot_bar_cos2(ca: prince.CA, df_ca: pd.crosstab, choice: str = "index", width=700, height=400):
    """
    Esta función genera el plot de los valores de cos2 para las primeras 2 componentes.

    :param ca: prince.CA
    :param df_ca: pd.DataFrame con quien se realizó el ajuste del objeto CA de la libreria prince.
    :param choice: Selección de filas ("index") o columnas ("columns) a plotear. Default "index".
    :param width: Ancho del plot.
    :param height:  Altura del plot.
    """
    if choice == "index":
        tmp = get_cos2(ca, df_ca, choice=choice)[["Dim1", "Dim2"]].sum(axis=1).sort_values(ascending=False)
        choice = "row"
    elif choice == "columns":
        tmp = get_cos2(ca, df_ca, choice=choice)[["Dim1", "Dim2"]].sum(axis=1).sort_values(ascending=False)

    fig = px.bar(x=tmp.index, y=tmp.values, template="plotly_white")
    fig.update_xaxes(title_text=" ")
    fig.update_yaxes(title_text="Cos2 - Quality of representation")
    fig.update_layout(width=width, height=height, title=f'Cos2 of {choice} to Dim-1-2')
    fig.show()


def plot_contrib(ca, df_ca, choice="index", axes=1, top=2):
    """
    Esta función plotea la contribución de cada perfil (fila/columna) en los ejes.
    """
    mapper_col_names = {0: "Dim1", 1: "Dim2", 2: "Dim3"}
    if choice == "index":
        contrib = ca.row_contributions_.rename(columns=mapper_col_names) * 100
        exp_value, choice = 100 / len(df_ca.index), "row"
    elif choice == "columns":
        contrib = ca.column_contributions_.rename(columns=mapper_col_names) * 100
        exp_value = 100 / len(df_ca.T.index)

    _, axes = plt.subplots(1, 2, constrained_layout=True, sharey=True)
    tmp1 = contrib.sort_values(by=["Dim1"], ascending=False)
    tmp2 = contrib.sort_values(by=["Dim2"], ascending=False)

    ax1 = sns.barplot(x=tmp1.index, y=tmp1["Dim1"], ax=axes[0], color="blue")
    ax1.set_xticklabels(labels=tmp1.index, rotation=-45)
    ax1.axhline(y=exp_value, color='black', ls='--', c="red")
    ax1.set_ylabel("Contributions (%)", fontsize=15, fontstyle="italic")
    ax1.set_title(f"Contribution of {choice} to Dim-1", fontsize=20, fontstyle="italic")

    ax2 = sns.barplot(x=tmp2.index, y=tmp2["Dim2"], ax=axes[1], color="blue")
    ax2.set_xticklabels(labels=tmp2.index, rotation=-45)
    ax2.axhline(y=exp_value, color='black', ls='--', c="red")
    ax2.set_ylabel("Contributions (%)", fontsize=15, fontstyle="italic")
    ax2.set_title(f"Contribution of {choice} to Dim-2", fontsize=20, fontstyle="italic")
    plt.show()


def plot_var_corr(ca: prince.CA, df, df_ca, dim1=1, dim2=2) -> None:
    """
    Plotea la correlación de las variables categóricas con cada una de las dimensiones.
    :param ca: Objeto CA de la librería prince.
    :param df: pd.DataFrame DataFrame con la tabla original de la información (sin get_dummies).
    :param df_ca: pd.DataFrame DataFrame utilizado para realizar el ajuste del objeto CA (obtenido con get_dummies).
    :param dim1: int Número de la primera dimensión a plotear (inicia desde 1).
    :param dim2: int Número de la segunda dimensión a plotear (inicia desde 1).
    :return: None
    """
    cos2, df_result = get_cos2(ca, df_ca, choice="columns"), pd.DataFrame()
    var_exp = np.round(np.array(ca.percentage_of_variance_), 2)
    for column in df.columns:
        tmp = [index for index in cos2.index if index.startswith(column)]
        a = cos2.loc[tmp].mean().to_frame(name=column)
        df_result[column] = cos2.loc[tmp].mean()
    data_plot = df_result.T.sort_values(by=[f"Dim{dim1}"])

    fig = px.scatter(
        data_plot, x=f"Dim{dim1}", y=f"Dim{dim2}", text=data_plot.index,
        template="plotly_white", symbol_sequence=['triangle-up'],
        color_discrete_sequence=["red"],
        title="Variables - MCA"
    )

    fig.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.update_traces(textposition=improve_text_position(data_plot[f"Dim{dim1}"]), textfont_color="red")
    fig.update_xaxes(title_text=f"Dim{dim1} ({var_exp[dim1 - 1]}%)")
    fig.update_yaxes(title_text=f"Dim{dim2} ({var_exp[dim2 - 1]}%)")
    fig.show()


def mca_biplot(ca: prince.CA, df_ca: pd.DataFrame, dim1: int = 1, dim2: int = 2, size: int = 11) -> None:
    """
    Esta función genera un biplot para un Análisis de Correspondencias Múltiples (MCA) a partir de un objeto CA
    de la librería prince. Incluye cos2.

    :param ca: Objeto CA (correspondencias múltiples) de la librería prince. Este objeto contiene las coordenadas
               y la varianza explicada por cada dimensión.
    :param df_ca: pd.DataFrame con los datos utilizados para ajustar el modelo de CA.
    :param dim1: int, opcional. Dimensión a representar en el eje X (por defecto 1).
    :param dim2: int, opcional. Dimensión a representar en el eje Y (por defecto 2).
    :param size: int, opcional. Tamaño del texto dentro del gráfico para evitar overlapping. Valor por defecto es 11.
    """
    row, col, title = ca.row_coordinates(df_ca), ca.column_coordinates(df_ca), "MCA-Biplot"
    columns_names = [f"Dim{i + 1}" for i in range(0, row.shape[1])]
    var_exp = np.round(np.array(ca.percentage_of_variance_), 2)
    row.columns, col.columns = columns_names, columns_names
    row["Options"], col["Options"], row["tmp"] = "show_rows_labels", "columns", "rows"
    row["cos2"] = get_cos2(ca, df_ca, choice="index")[[f"Dim{dim1}", f"Dim{dim2}"]].sum(axis=1)
    col["cos2"] = get_cos2(ca, df_ca, choice="columns")[[f"Dim{dim1}", f"Dim{dim2}"]].sum(axis=1)
    row = row.sort_values(by=[f"Dim{dim1}"]).round(3)
    row1 = row.reset_index().groupby(by=[f"Dim{dim1}"], as_index=False).agg({"index": lambda x: list(x)})
    row, row.index = row.drop_duplicates(subset=[f"Dim{dim1}"]), row1["index"]
    data_plot = pd.concat([row, col]).sort_values(by=[f"Dim{dim1}"])
    minx, miny = min(data_plot[f"Dim{dim1}"]) - 0.35, min(data_plot[f"Dim{dim2}"]) - 0.35,
    maxx, maxy = max(data_plot[f"Dim{dim1}"]) + 0.35, max(data_plot[f"Dim{dim2}"]) + 0.35

    fig = px.scatter(
        data_plot.round(4), x=f"Dim{dim1}", y=f"Dim{dim2}", color="cos2",
        color_discrete_sequence=["red", "slateblue"],
        color_continuous_scale="temps",
        symbol="Options", symbol_sequence=["triangle-up", "circle"],
        text=data_plot.index,
        template="plotly_white",
        title=title,
        hover_data={"Options": False},
        hover_name=data_plot.index
    )

    fig.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.update_xaxes(range=[minx, maxx], title_text=f"Dim{dim1} ({var_exp[dim1 - 1]}%)")
    fig.update_yaxes(range=[miny, maxy], title_text=f"Dim{dim2} ({var_exp[dim2 - 1]}%)")
    fig.update_traces(textposition=improve_text_position(data_plot[f"Dim{dim1}"]), )
    fig.update_layout(font_size=size)
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            x=1.1,
            y=0.4,
            len=0.5,
            title="cos²",
            thickness=20,
            xpad=10
        ),
        legend=dict(
            yanchor="top",
            x=1.1,
            y=1
        )
    ).show()
    

def mca_plot_cos2(ca, df_ca, choice="index", axes1=1, axes2=2, top=5, size=11):
    """
    Esta función plotea el cos2 de un análisis de correspondencias (CA) o de un análisis de 
    correspondencias múltiples (CMA).
    Input:----> ca: Objeto CA de la libreria prince.
          ----> df_ca: pd.DataFrame con quien se realizó el ajuste del objeto CA de la libreria prince.
                      Este debe ser el obtenido con get_dummies.
          ----> choice (str): Selección de filas ("index") o columnas ("columns) a plotear. 
                              Default "index".
          ----> axes1, axes2 (int): Números enteros de las dimensiones que se desean plotear para ver
                                    la calidad de representación de la modalidad en las dimensiones.
          ----> top (int): Cantidad de variables a mostar en el plot.
          ----> size (int +): Tamaño del texto del plot para buscar evitar overlapping
    """
    var_exp = np.round(np.array(ca.percentage_of_variance_), 2)

    if choice == "index":
        tmp, symbol = ca.row_coordinates(df_ca), ["circle"]
        tmp["cos2"] = get_cos2(ca, df_ca, choice=choice)[[f"Dim{axes1}", f"Dim{axes2}"]].sum(axis=1)
        tmp = tmp.sort_values(by=0).round(3)
        tmp1 = tmp.reset_index().groupby(by=0, as_index=False).agg({"index": lambda x: list(x)})
        tmp = tmp.drop_duplicates(subset=[0])
        tmp.index = tmp1["index"]
    elif choice == "columns":
        tmp, symbol = ca.column_coordinates(df_ca), ["triangle-up"]
        tmp["cos2"] = get_cos2(ca, df_ca, choice=choice)[[f"Dim{axes1}", f"Dim{axes2}"]].sum(axis=1)
    tmp.columns = [f"Dim{i + 1}" for i in range(0, tmp.shape[1] - 1)] + ["cos2"]
    minx, maxx = min(tmp[f"Dim{axes1}"]) - 0.3, max(tmp[f"Dim{axes1}"]) + 0.3
    miny, maxy = min(tmp[f"Dim{axes2}"]) - 0.3, max(tmp[f"Dim{axes2}"]) + 0.3
    tmp = tmp.sort_values(by=[f"Dim{axes1}"]).round(4).head(top)

    fig = px.scatter(data_frame=tmp, x=f"Dim{axes1}", y=f"Dim{axes2}", text=tmp.index,
                     template="plotly_white", color="cos2", color_continuous_scale="temps",
                     symbol_sequence=symbol, )
    if choice == "index":
        fig.update_layout(title="Row points - MCA")
    elif choice == "columns":
        fig.update_layout(title="Column points - MCA")
    fig.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.update_traces(textposition=improve_text_position(tmp[f"Dim{axes1}"]))
    fig.update_xaxes(range=[minx, maxx], title_text=f"Dim{axes1} ({var_exp[axes1 - 1]}%)")
    fig.update_yaxes(range=[miny, maxy], title_text=f"Dim{axes2} ({var_exp[axes2 - 1]}%)")
    fig.update_layout(font_size=size)
    fig.show()


def plot_contrib_mca(ca, df_ca, choice="index", axes1=1, axes2=2, top=5) -> None:
    """
    Plotea la contribución de cada perfil en los ejes seleccionados.

    :param ca: Objeto CA de la librería prince.
    :param df_ca: pd.DataFrame con quien se realizó el ajuste del objeto CA. Debe ser el obtenido con get_dummies.
    :param choice: Selección de filas ("index") o columnas ("columns") a plotear. Default: "index".
    :param axes1: Número entero de la primera dimensión a plotear.
    :param axes2: Número entero de la segunda dimensión a plotear.
    :param top: Cantidad de variables a mostrar en el plot.
    """
    columns = [f"Dim{i + 1}" for i in range(0, ca.column_contributions_.shape[1])]

    if choice == "index":
        contrib = ca.row_contributions_.rename(columns=columns) * 100
        exp_value, choice = 100 / len(df_ca.index), "row"

    elif choice == "columns":
        contrib, contrib.columns, = ca.column_contributions_ * 100, columns
        exp_value, choice = 100 / len(df_ca.T.index), "variables"

    tmp1 = contrib.sort_values(by=[f"Dim{axes1}"], ascending=False).head(top)
    tmp2 = contrib.sort_values(by=[f"Dim{axes2}"], ascending=False).head(top)
    tmp, axess, indexs = [tmp1, tmp2], [axes1, axes2], [0, 1]
    zipper = zip(tmp, axess, indexs)

    _, axes = plt.subplots(1, 2, constrained_layout=True, sharey=True)
    for i in zipper:
        ax = sns.barplot(x=i[0].index, y=i[0][f"Dim{i[1]}"], ax=axes[i[2]], color="blue")
        ax.set_xticklabels(labels=i[0].index, rotation=90)
        ax.axhline(y=exp_value, color="black", ls="--", c="red")
        ax.set_ylabel("Contributions (%)", fontsize=15, fontstyle="italic")
        ax.set_title(f"Contribution of {choice} to Dim-{i[1]}", fontsize=20, fontstyle="italic")
    plt.show()


def mca_screeplot(ca, figsize=(15, 6)):
    """
    Esta función genera el gráfico de sedimentación del análisis de correspondencias.
    Input: ----> ca: Objeto CA de la libreria prince.
          ----> figsize: Tupla (x,y); x: ancho del plot, y: alto del plot.
    Output: ---> None.
    """
    inertia = ca.percentage_of_variance_
    barras, n_compo = np.array(inertia), len(inertia)
    plt.figure(figsize=figsize)
    plt.bar([str(i + 1) for i in range(0, n_compo)], barras, color="blue")
    plt.xlabel("Dimensions")
    plt.ylabel("Percentage of explained variance")
    plt.plot([str(i + 1) for i in range(0, n_compo)], barras, color="black", marker="o")
    plt.title("Scree plot.", fontsize=20, loc="left")
    plt.show()
    
    
def mca_triplot(mca: prince.CA, df_mca: pd.DataFrame, width: int = 1000, height: int = 800, delta: float = 0.5):
    """
    Genera un triplot 3D para visualizar la proyección de las variables en las primeras 3 dimensiones del análisis de correspondencias.

    :param mca: prince.CA Objeto CA de la librería prince.
    :param df_mca: pd.DataFrame DataFrame utilizado para realizar el ajuste del objeto CA.
    :param width: int, opcional Ancho del gráfico (default: 1000).
    :param height: int, opcional Alto del gráfico (default: 800).
    :param delta: float, opcional Delta para ajustar los límites de los ejes (default: 0.5).
    :return: None
    """
    df = mca.column_coordinates(df_mca)
    df.columns = [f"Dim{i+1}" for i in range(len(df.columns))]
    df["cos2"] = get_cos2(mca, df_mca, choice="columns")[["Dim1", "Dim2", "Dim3"]].sum(axis=1)
    df = df.round(4)

    x_min, x_max = df["Dim1"].min() - delta, df["Dim1"].max() + delta
    y_min, y_max = df["Dim2"].min() - delta, df["Dim2"].max() + delta
    z_min, z_max = df["Dim3"].min() - delta, df["Dim3"].max() + delta

    go.Figure(data=[go.Scatter3d(
            x=df["Dim1"],
            y=df["Dim2"],
            z=df["Dim3"],
            text=df.index,
            mode="markers+text",
            marker=dict(
                size=10,
                color=df["cos2"],
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(
                    title="cos²",
                    thickness=30,
                    len=1,
                    yanchor="middle",
                    y=0.5
                )
            ),
            textposition="top center",
            hovertemplate=
            "<b>%{text}</b><br><br>" +
            "Dim 1: %{x}<br>" +
            "Dim 2: %{y}<br>" +
            "Dim 3: %{z}<br>" +
            "cos²: %{customdata[0]:.4f}<br>" +
            "<extra></extra>",
            customdata=df[["cos2"]]
            )
        ]
        ).update_layout(
        scene=dict(
            xaxis=dict(title="Dim 1", range=[x_min, x_max]),
            yaxis=dict(title="Dim 2", range=[y_min, y_max]),
            zaxis=dict(title="Dim 3", range=[z_min, z_max])
        ),
        title="Triplot: Proyección en las 3 primeras dimensiones",
        showlegend=False,
        width=width,
        height=height 
    ).show()
