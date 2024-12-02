import pandas as pd
import numpy as np


def dataframes_example_ca():
    """
    Dataframes para ejemplo 1 del analisis de correspondencias simples.
    """
    data = np.array(
        [688, 116, 584, 188, 4, 326, 38, 241, 110, 3, 343, 84, 909, 412, 26, 98, 48, 403, 681, 85]
    ).reshape(4, 5)
    data_fi = 100 * data / np.sum(data)

    columns = ["Rubio(ru)", "Rojo(r)", "Medio(m)", "Oscuro(o)", "Negro(n)"]
    index = ["Claros(C)", "Azules(A)", "Medio(M)", "Oscuros(O)"]
    example_df = pd.DataFrame(data, columns=columns, index=index)
    example_df["Total_(fi.)"] = example_df.apply(np.sum, axis=1)
    example_df.loc["Total_(f.j)"] = example_df.apply(np.sum, axis=0)
    example_df.columns.rename('Color Cabello', inplace=True)
    example_df.index.rename('Color Ojos', inplace=True)

    example_dfi = pd.DataFrame(
        data_fi, columns=["Rubio(ru)", "Rojo(r)", "Medio(m)", "Oscuro(o)", "Negro(n)"],
        index=["Claros(C)", "Azules(A)", "Medio(M)", "Oscuros(O)"]
    )

    example_dfi["Total_(fi.)"] = example_dfi.apply(np.sum, axis=1)
    example_dfi.loc["Total_(f.j)"] = example_dfi.apply(np.sum, axis=0)
    example_dfi.loc["Total_(f.j)", "Total_(fi.)"] = 100

    # Perfil fila.
    pf_row = example_df.copy()
    for column in pf_row.columns:
        pf_row[column] = pf_row[column] / pf_row["Total_(fi.)"]

    # Perfil columna.
    pf_col = example_df.T.copy()
    for column in pf_col.columns:
        pf_col[column] = pf_col[column] / pf_col["Total_(f.j)"]

    return data, example_df, example_dfi, pf_row, pf_col
