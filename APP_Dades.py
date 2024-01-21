import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import base64
from streamlit_option_menu import option_menu
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
import matplotlib.colors as colors
from datetime import datetime
import plotly.graph_objs as go
import json
import io

path = ""

st.set_page_config(
    page_title="Viabilidad de una promoción",
    page_icon="""data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAA1VBMVEVHcEylpKR6eHaBgH9GREGenJxRT06op6evra2Qj49kYWCbmpqdnJyWlJS+vb1CPzyurKyHhYWMiYl7eXgOCgiPjY10cnJZV1WEgoKCgYB9fXt
    /fHyzsrGUk5OTkZGlo6ONioqko6OLioq7urqysbGdnJuurazCwcHLysp+fHx9fHuDgYGJh4Y4NTJcWVl9e3uqqalcWlgpJyacm5q7urrJyMizsrLS0tKIhoaMioqZmJiTkpKgn5+Bf36WlZWdnJuFg4O4t7e2tbXFxMR3dXTg39/T0dLqKxxpAAAAOHRSTlMA/WCvR6hq/
    v7+OD3U9/1Fpw+SlxynxXWZ8yLp+IDo2ufp9s3oUPII+jyiwdZ1vczEli7waWKEmIInp28AAADMSURBVBiVNczXcsIwEAVQyQZLMrYhQOjV1DRKAomKJRkZ+P9PYpCcfbgze+buAgDA5nf1zL8TcLNamssiPG/
    vt2XbwmA8Rykqton/XVZAbYKTSxzVyvVlPMc4no2KYhFaePvU8fDHmGT93i47Xh8ijPrB/0lTcA3lcGQO7otPmZJfgwhhoytPeKX5LqxOPA9i7oDlwYwJ3p0iYaEqWDdlRB2nkDjgJPA7nX0QaVq3kPGPZq/V6qUqt9BAmVaCUcqEdACzTBFCpcyvFfAAxgMYYVy1sTwAAAAASUVORK5CYII=""",
    layout="wide"
)
def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file(path + "main.css")
st.subheader("**ESTUDIO DE VIABILIDAD DE UNA PROMOCIÓN**")
left_col, right_col, margin_right = st.columns((0.15, 1, 0.15))
with right_col:
    selected = option_menu(
        menu_title=None,  # required
        options=["Análisis estático","Análisis dinámico","Ratios financieras", "Análisis de mercado", "Resumen de resultados"],  # Dropdown menu
        icons=[None, None, None, None],  # Icons for dropdown menu
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0px important!", "background-color": "#fcefdc", "align":"center", "overflow":"hidden"},
            "icon": {"color": "#bf6002", "font-size": "17px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "center",
                "font-weight": "bold",
                "color":"#363534",
                "padding": "5px",
                "--hover-color": "#fcefdc",
                "background-color": "#fcefdc",
                "overflow":"hidden"},
            "nav-link-selected": {"background-color": "#de7207"}
            })
if selected == "Análisis estático":
    left, right= st.columns((1,1))
    with left:
        st.markdown(
            """
            <style>
                .title-box {
                    border: 2px solid red;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                    color: red;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<h1 class="title-box">GASTOS</h1>', unsafe_allow_html=True)
        st.header("SOLAR")
        # st.write("Coste asociado al solar es", user_input)
        input_solar1 = st.number_input("**COMPRA DEL SOLAR**", min_value=0, max_value=999999999, value=0, step=1000)
        input_solar2 = st.number_input("**IMPUESTOS (AJD...)**", min_value=0, max_value=999999999, value=0, step=1000)
        input_solar3 = st.number_input("**NOTARIA, REGISTRO, COMISIONES...**", min_value=0, max_value=999999999, value=0, step=1000)
        st.header("EDIFICACIÓN")
        input_edificacion2 = st.number_input("**HONORARIOS PROFESIONALES**", min_value=0, max_value=999999999, value=0, step=1000)
        input_edificacion3 = st.number_input("**IMPUESTOS Y TASAS MUNICIPALES**", min_value=0, max_value=999999999, value=0, step=1000)
        input_edificacion4 = st.number_input("**ACOMETIDAS**", min_value=0, max_value=999999999, value=0, step=1000)
        input_edificacion5 = st.number_input("**CONSTRUCCIÓN**", min_value=0, max_value=999999999, value=0, step=1000)
        input_edificacion6 = st.number_input("**POSTVENTA**", min_value=0, max_value=999999999, value=0, step=1000)
        st.header("COMERCIALIZACIÓN")
        input_com1 = st.number_input("**COMISIONES (5% VENDA)**", min_value=0, max_value=999999999, value=0, step=1000)
        st.header("ADMINISTRACIÓN")
        input_admin1 = st.number_input("**GASTOS DE ADMINISTRACIÓN**", min_value=0, max_value=999999999, value=0, step=1000)
        total_gastos = input_solar1 + input_solar2 + input_solar3 + input_edificacion2 + input_edificacion3 + input_edificacion4 + input_edificacion5 + input_edificacion6
        st.metric(label="**TOTAL GASTOS**", value=total_gastos)
    with right:
        st.markdown(
            """
            <style>
                .title-box-ing {
                    border: 2px solid blue;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                    color: blue;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<h1 class="title-box-ing">INGRESOS</h1>', unsafe_allow_html=True)
        st.header("VENTAS")
        input_ventas1 = st.number_input("**INGRESOS POR VENTAS**", min_value=0, max_value=999999999, value=0, step=1000)
        total_ingresos = input_ventas1 + 0
        st.metric(label="**TOTAL INGRESOS**", value=total_ingresos)
        st.markdown(
            """
            <style>
                .title-box-fin {
                    border: 2px solid #c9b00e;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                    color: #c9b00e;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<h1 class="title-box-fin">FINANCIACIÓN</h1>', unsafe_allow_html=True)
        input_fin1 = st.number_input("**INTERESES HIPOTECA**", min_value=0, max_value=999999999, value=0, step=1000)
        input_fin2 = st.number_input("**GASTOS DE CONSTITUCIÓN**", min_value=0, max_value=999999999, value=0, step=1000)
        total_fin = input_fin1 + input_fin2
        st.metric(label="**TOTAL GASTOS DE FINANCIACIÓN**", value=total_fin)
        st.markdown(
            """
            <style>
                .title-box-res {
                    border: 2px solid grey;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                    color: grey;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<h1 class="title-box-res">RESULTADO ANTES DE IMPUESTOS E INTERESES (BAII)</h1>', unsafe_allow_html=True)
        st.metric(label="**BAII**", value=total_ingresos - total_gastos)
        st.markdown('<h1 class="title-box-res">RESULTADO ANTES DE IMPUESTOS (BAI)</h1>', unsafe_allow_html=True)
        st.metric(label="**BAI**", value=total_ingresos - total_gastos - total_fin)
@st.cache_resource
def import_data():
    maestro_mun = pd.read_excel("Maestro_MUN_COM_PROV.xlsx", sheet_name="Maestro")
    DT_mun_def = pd.read_excel("DT_simple.xlsx", sheet_name="mun_q")
    DT_mun_y_def = pd.read_excel("DT_simple.xlsx", sheet_name="mun_y")
    return([DT_mun_def, DT_mun_y_def, maestro_mun])

DT_mun, DT_mun_y, maestro_mun = import_data()

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_m(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Month"] = output_data['Fecha'].dt.month
    output_data = output_data.dropna()
    output_data = output_data[(output_data["Month"]<=output_data['Month'].iloc[-1])]
    return(output_data.drop(["Data", "Month"], axis=1))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Trimestre"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Trimestre"] + columns_output

    return(output_data.set_index("Trimestre").drop("Data", axis=1))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_anual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = columns_output
    output_data["Any"] = output_data["Any"].astype(str)
    return(output_data.set_index("Any"))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_mensual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Fecha"] = output_data["Fecha"].astype(str)
    return(output_data.set_index("Fecha"))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present(data_ori, columns_sel, year):
    output_data = data_ori[data_ori[columns_sel]!=0][["Trimestre"] + [columns_sel]].dropna()
    output_data["Trimestre_aux"] = output_data["Trimestre"].str[-1]
    output_data = output_data[(output_data["Trimestre_aux"]<=output_data['Trimestre_aux'].iloc[-1])]
    output_data["Any"] = output_data["Trimestre"].str[0:4]
    output_data = output_data.drop(["Trimestre", "Trimestre_aux"], axis=1)
    output_data = output_data.groupby("Any").mean().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==str(year)]
    output_data = output_data.set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + [columns_sel]]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha", columns_sel])
    output_data = output_data.groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly_aux(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly_diff(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.groupby("Any").mean().diff().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def indicator_year(df, df_aux, year, variable, tipus, frequency=None):
    if (year==str(datetime.now().year-1) and (frequency=="month") and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present_monthly(df_aux, variable, year),2))
    if (year==str(datetime.now().year-1) and (frequency=="month_aux") and (tipus=="var")):
        return(round(tidy_present_monthly_aux(df_aux, variable, year),2))
    if (year==str(datetime.now().year-1) and (frequency=="month_aux") and ((tipus=="diff"))):
        return(round(tidy_present_monthly_diff(df_aux, variable, year),2))
    if (year==str(datetime.now().year-1) and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present(df_aux.reset_index(), variable, year),2))
    if tipus=="level":
        df = df[df.index==year][variable]
        return(round(df.values[0],2))
    if tipus=="var":
        df = df[variable].pct_change().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2))
    if tipus=="diff":
        df = df[variable].diff().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def concatenate_lists(list1, list2):
    result_list = []
    for i in list1:
        result_element = i+ list2
        result_list.append(result_element)
    return(result_list)


def filedownload(df, filename):
    towrite = io.BytesIO()
    df.to_excel(towrite, encoding='latin-1', index=True, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode("latin-1")
    href = f"""<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">
    <button class="download-button">Descarregar</button></a>"""
    return href

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def line_plotly(table_n, selection_n, title_main, title_y, title_x="Trimestre", replace_0=False):
    plot_cat = table_n[selection_n]
    if replace_0==True:
        plot_cat = plot_cat.replace(0, np.NaN)
    colors = ['#2d538f', '#de7207', '#385723']
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Scatter(
            x=plot_cat.index,
            y=plot_cat[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title=title_x),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ['#2d538f', '#de7207', '#385723']
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Any"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig
#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def stacked_bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ['#2d538f', '#de7207', '#385723']
    
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Any"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        barmode='stack',
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig
#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def area_plotly(table_n, selection_n, title_main, title_y, trim):
    plot_cat = table_n[table_n.index>=trim][selection_n]
    fig = px.area(plot_cat, x=plot_cat.index, y=plot_cat.columns, title=title_main)
    fig.for_each_trace(lambda trace: trace.update(fillcolor = trace.line.color))
    fig.update_layout(xaxis_title="Trimestre", yaxis=dict(title=title_y, tickformat=",d"), barmode='stack')
    fig.update_traces(opacity=0.4)  # Change opacity to 0.8
    fig.update_layout(legend_title_text="")
    fig.update_layout(
        title=dict(text=title_main, font=dict(size=13), y=0.97),
        legend=dict(x=-0.15, y=1.25, orientation="h"),  # Adjust the x and y values for the legend position
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    return fig

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def table_monthly(data_ori, year_ini, rounded=True):
    data_ori = data_ori.reset_index()
    month_mapping_catalan = {
        1: 'Gener',
        2: 'Febrer',
        3: 'Març',
        4: 'Abril',
        5: 'Maig',
        6: 'Juny',
        7: 'Juliol',
        8: 'Agost',
        9: 'Setembre',
        10: 'Octubre',
        11: 'Novembre',
        12: 'Desembre'
    }

    try:
        output_data = data_ori[data_ori["Data"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Data'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = output_data[numeric_columns].applymap(lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "Data"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    except KeyError:
        output_data = data_ori[data_ori["Fecha"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Fecha'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = output_data[numeric_columns].applymap(lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "index"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    return(output_data)

def format_dataframes(df, style_n):
    if style_n==True:
        return(df.style.format("{:,.0f}"))
    else:
        return(df.style.format("{:,.1f}"))



def table_trim(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    data_ori["Any"] = data_ori["Trimestre"].str.split("T").str[0]
    data_ori["Trimestre"] = data_ori["Trimestre"].str.split("T").str[1]
    data_ori["Trimestre"] = data_ori["Trimestre"] + "T"
    data_ori = data_ori[data_ori["Any"]>=str(year_ini)]
    data_ori = data_ori.replace(0, np.NaN)
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = data_ori[numeric_columns].applymap(lambda x: round(x, 1))
    output_data = data_ori.set_index(["Any", "Trimestre"]).T.dropna(axis=1, how="all")
    if formated==True:   
        return(format_dataframes(output_data, True))
    else:
        return(format_dataframes(output_data, False))


def table_year(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = data_ori[numeric_columns].applymap(lambda x: round(x, 1))
    data_output = data_ori[data_ori["Any"]>=str(year_ini)].T
    data_output.columns = data_output.iloc[0,:]
    data_output = data_output.iloc[1:,:]
    if formated==True:   
        return(format_dataframes(data_output, True))
    else:
        return(format_dataframes(data_output, False))

if selected=="Análisis de mercado":
    left, center, right = st.columns((1,1,1))
    with left:
        selected_type = st.radio("**Mercado de venta o alquiler**", ("Venta", "Alquiler"))
    with center:
        selected_mun = st.selectbox("**Selecciona un municipio:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"))
    with right:
        max_year=datetime.now().year-1
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.selectbox("**Selecciona un año:**", available_years, available_years.index(2023))
    if selected_type=="Venta":
        min_year=2014
        st.subheader(f"PRECIOS PER M\u00b2 CONSTRUIDO EN {selected_mun.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)   
        table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Precio de la vivienda total", "Precio de la vivienda de segunda mano", "Precio de la vivienda nueva"])
        table_mun = table_mun.replace(0, np.NaN)
        table_mun_y = table_mun.reset_index().copy()
        table_mun_y["Any"] = table_mun_y["Trimestre"].str[:4]
        table_mun_y = table_mun_y.drop("Trimestre", axis=1)
        table_mun_y = table_mun_y.groupby("Any").mean()
        left, center, right = st.columns((1,1,1))
        with left:
            try:
                st.metric(label="**Precio de la vivienda total** (€/m\u00b2 construido)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Precio de la vivienda total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Precio de la vivienda total", "var")}%""")
            except IndexError:
                st.metric(label="**Precio de la vivienda total** (€/m\u00b2 construido)", value="n/a") 
        with center:
            try:
                st.metric(label="**Precio de la vivienda de segunda mano** (€/m\u00b2 construido)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Precio de la vivienda de segunda mano", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Precio de la vivienda de segunda mano", "var")}%""")
            except IndexError:
                st.metric(label="**Precio de la vivienda de segunda mano** (€/m\u00b2 construido)", value="n/a") 
        with right:
            try:
                st.metric(label="**Precio de la vivienda nueva** (€/m\u00b2 construido)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Precio de la vivienda nueva", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Precio de la vivienda nueva", "var")}%""") 
            except IndexError:
                st.metric(label="**Precio de la vivienda nueva** (€/m\u00b2 construido)", value="n/a") 
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_mun, 2020, True, False).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"Preus_{selected_mun}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"Preus_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolución trimestral dels preus per m\u00b2 construido por tipologia", "€/m\u00b2 útil", True), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolución anual dels preus per m\u00b2 construido por tipologia", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
    if selected_type=="Alquiler":
        min_year=2014
        st.subheader(f"MERCAT DE LLOGUER A {selected_mun.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Nombre de contractes de lloguer**", value="n/a")
        with right_col:
            try:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="n/a")
                st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_mun, 2020, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_mun, 2014, rounded=True), f"{selected_type}_{selected_mun}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_mun_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=True), f"{selected_type}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_mun, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", True), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_mun, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_mun_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_mun_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)