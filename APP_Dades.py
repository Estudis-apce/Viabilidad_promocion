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
from io import StringIO
import io
from streamlit_gsheets import GSheetsConnection
import json
import geopandas as gpd

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

selected = option_menu(
    menu_title=None,  # required
    options=["Edificabilidad","Análisis estático","Análisis dinámico", "Análisis de mercado", "Análisis de mercado APCE", "Resumen de resultados"],  # Dropdown menu
    icons=[None, None, None, None],  # Icons for dropdown menu
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0px important!", "background-color": "white", "align":"center", "overflow":"hidden"},
        "icon": {"color": "#def3fa", "font-size": "17px"},
        "nav-link": {
            "font-size": "17px",
            "text-align": "center",
            "font-weight": "bold",
            "color":"black",
            "padding": "5px",
            "--hover-color": "#ADD8E6",
            "background-color": "white",
            "overflow":"hidden"},
        "nav-link-selected": {"background-color": "#ADD8E6"}
        })
if selected=="Edificabilidad":
#     left, right= st.columns((0.5,2))
#     with left:
#         uploaded_file = st.file_uploader("**Cargar archivo con los parametros de edificabilidad**", type="xlsx")
#     with right:
#         if uploaded_file is not None:
#             params_edif = pd.read_excel(uploaded_file)
#             st.markdown(params_edif.to_html(), unsafe_allow_html=True)
    conn = st.connection("gsheets", type=GSheetsConnection)
    existing_data = conn.read(worksheet="Edificabilidad", usecols=list(range(3)), ttl=5, encoding="ISO-8859-1")
    existing_data = existing_data.dropna(how="all")
    left, right = st.columns((1,1))
    for index, row in existing_data.iterrows():
        if index % 2 == 0:
            with left:
                input_value = st.number_input(f"**{row['Elemento']}:**", value=row['Superficie'])
                exec(f"{row['Nombre']} = {input_value}")
        else:
            with right:
                input_value = st.number_input(f"**{row['Elemento']}:**", value=row['Superficie'])
                exec(f"{row['Nombre']} = {input_value}")
    submit_button = st.button(label="Guardar proposta")
    result_df = pd.DataFrame(columns=['Nombre', 'Superficie'])
    if submit_button:
        for variable_name in existing_data['Nombre']:
            variable_value = locals().get(variable_name, None)  # Get the value of the variable by name
            result_df = result_df.append({'Nombre': variable_name, 'Superficie': variable_value}, ignore_index=True)
        result_df = pd.concat([existing_data.iloc[:,0], result_df], axis=1)
        conn.update(worksheet="Edificabilidad", data=result_df)
        st.success("¡Propuesta guardada!")


if selected == "Análisis estático":
    left, center, right= st.columns((1,1,1))
    with center:
        selected_propuesta = st.radio("", ("Propuesta 1", "Propuesta 2", "Propuesta 3"), horizontal=True)
    conn = st.connection("gsheets", type=GSheetsConnection)
    existing_data = conn.read(worksheet=selected_propuesta, usecols=list(range(3)), ttl=5)
    existing_data = existing_data.dropna(how="all")

    for index, row in existing_data.iterrows():
        variable_name = row['Nombre']
        variable_value = (row['Valor'])
        exec(f"{variable_name} = {variable_value}")
    
    left, right= st.columns((1,1))
    with left:
        st.markdown('<h1 class="title-box">GASTOS</h1>', unsafe_allow_html=True)
        st.header("SOLAR")
        # st.write("Coste asociado al solar es", user_input)
        input_solar1 = st.number_input("**COMPRA DEL SOLAR**", min_value=0.0, max_value=999999999.0, value=input_solar1, step=1000.0)
        input_solar2 = st.number_input("**IMPUESTOS (AJD...)**", min_value=0.0, max_value=999999999.0, value=input_solar2, step=1000.0)
        input_solar3 = st.number_input("**NOTARIA, REGISTRO, COMISIONES...**", min_value=0.0, max_value=999999999.0, value=input_solar3, step=1000.0)
        st.header("EDIFICACIÓN")
        input_edificacion2 = st.number_input("**HONORARIOS PROFESIONALES**", min_value=0.0, max_value=999999999.0, value=input_edificacion2, step=1000.0)
        input_edificacion3 = st.number_input("**IMPUESTOS Y TASAS MUNICIPALES**", min_value=0.0, max_value=999999999.0, value=input_edificacion3, step=1000.0)
        input_edificacion4 = st.number_input("**ACOMETIDAS**", min_value=0.0, max_value=999999999.0, value=input_edificacion4, step=1000.0)
        input_edificacion5 = st.number_input("**CONSTRUCCIÓN**", min_value=0.0, max_value=999999999.0, value=input_edificacion5, step=1000.0)
        input_edificacion6 = st.number_input("**POSTVENTA**", min_value=0.0, max_value=999999999.0, value=input_edificacion6, step=1000.0)
        st.header("COMERCIALIZACIÓN")
        input_com1 = st.number_input("**COMISIONES (5% VENDA)**", min_value=0.0, max_value=999999999.0, value=input_com1, step=1000.0)
        st.header("ADMINISTRACIÓN")
        input_admin1 = st.number_input("**GASTOS DE ADMINISTRACIÓN**", min_value=0.0, max_value=999999999.0, value=input_admin1, step=1000.0)
        total_gastos = input_solar1 + input_solar2 + input_solar3 + input_edificacion2 + input_edificacion3 + input_edificacion4 + input_edificacion5 + input_edificacion6
        st.metric(label="**TOTAL GASTOS**", value=total_gastos)
    with right:
        st.markdown('<h1 class="title-box-ing">INGRESOS</h1>', unsafe_allow_html=True)
        st.header("VENTAS")
        input_ventas1 = st.number_input("**INGRESOS POR VENTAS**", min_value=0.0, max_value=999999999.0, value=input_ventas1, step=1000.0)
        total_ingresos = input_ventas1 + 0
        st.metric(label="**TOTAL INGRESOS**", value=total_ingresos)
        st.markdown('<h1 class="title-box-fin">FINANCIACIÓN</h1>', unsafe_allow_html=True)
        input_fin1 = st.number_input("**INTERESES HIPOTECA**", min_value=0.0, max_value=999999999.0, value=input_fin1, step=1000.0)
        input_fin2 = st.number_input("**GASTOS DE CONSTITUCIÓN**", min_value=0.0, max_value=999999999.0, value=input_fin2, step=1000.0)
        total_fin = input_fin1 + input_fin2
        st.metric(label="**TOTAL GASTOS DE FINANCIACIÓN**", value=total_fin)
        st.markdown('<h1 class="title-box-res">RESULTADO ANTES DE IMPUESTOS E INTERESES (BAII)</h1>', unsafe_allow_html=True)
        st.metric(label="**BAII**", value=total_ingresos - total_gastos)
        st.markdown('<h1 class="title-box-res">RESULTADO ANTES DE IMPUESTOS (BAI)</h1>', unsafe_allow_html=True)
        st.metric(label="**BAI**", value=total_ingresos - total_gastos - total_fin)
        
        submit_button = st.button(label="Guardar proposta")
        result_df = pd.DataFrame(columns=['Nombre', 'Valor'])
        if submit_button:
            for variable_name in existing_data['Nombre']:
                variable_value = locals().get(variable_name, None)  # Get the value of the variable by name
                result_df = result_df.append({'Nombre': variable_name, 'Valor': variable_value}, ignore_index=True)
            result_df = pd.concat([existing_data.iloc[:,0], result_df], axis=1)
            conn.update(worksheet=selected_propuesta, data=result_df)
            st.success("¡Propuesta guardada!")
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing_data = conn.read(worksheet=selected_propuesta, usecols=list(range(3)), ttl=5)
        existing_data = existing_data.dropna(how="all")
        with left:
            def treemap():
                # Create the treemap
                fig = px.treemap(
                    existing_data[["Parametro", "Valor"]][existing_data["Parametro"]!="INGRESOS POR VENTAS"],
                    names='Parametro',
                    values='Valor',
                    parents=[''] * len(existing_data[["Parametro", "Valor"]][existing_data["Parametro"]!="INGRESOS POR VENTAS"]),
                    title="Distribució dels costos de la promoció",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=600, width=1750, paper_bgcolor = "#fcefdc", plot_bgcolor='#fcefdc')
                return fig

            st.plotly_chart(treemap())
if selected == "Análisis dinámico":
    left_c, left, center, right, right_c = st.columns((1,1,1,1,1))
    with left:
        start_date =st.date_input("Fecha de inicio de la operación", value="today")
    with center:
        max_trim = st.slider("**Número de trimestres de la operación**", 8, 24, 8)
    with right:
        selected_propuesta = st.radio("", ("Propuesta 1", "Propuesta 2", "Propuesta 3"), horizontal=True)
    

################################# ANALISI DE MERCADO DATOS PÚBLICOS (AHC) #########################################################

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

############################################################# PESTÑA ANALISIS DE MERCADO ########################################################

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


############################################################# PROVINCIES FUNCIONS #############################################
@st.cache_resource
def table_geo(geo, any_ini, any_fin, selected):
    if selected=="Àmbits territorials":
        df_prov_filtered = ambits_df[(ambits_df["GEO"]==geo) & (ambits_df["Any"]>=any_ini) & (ambits_df["Any"]<=any_fin)].pivot(index=["Any"], columns=["Tipologia", "Variable"], values="Valor")
        df_prov_n = df_prov_filtered.sort_index(axis=1, level=[0,1])
        num_cols = df_prov_n.select_dtypes(include=['float64', 'int64']).columns
        df_prov_n[num_cols] = df_prov_n[num_cols].round(0)
        df_prov_n[num_cols] = df_prov_n[num_cols].astype("float64")
        num_cols = df_prov_n.select_dtypes(include=['float64', 'int']).columns
        df_prov_n[num_cols] = df_prov_n[num_cols].applymap(lambda x: '{:,.0f}'.format(x).replace(',', '#').replace('.', ',').replace('#', '.'))
        return(df_prov_n)
    if selected=="Províncies" or selected=="Catalunya":
        df_prov_filtered = provincia_df[(provincia_df["GEO"]==geo) & (provincia_df["Any"]>=any_ini) & (provincia_df["Any"]<=any_fin)].pivot(index=["Any"], columns=["Tipologia", "Variable"], values="Valor")
        df_prov_n = df_prov_filtered.sort_index(axis=1, level=[0,1])
        num_cols = df_prov_n.select_dtypes(include=['float64', 'int64']).columns
        df_prov_n[num_cols] = df_prov_n[num_cols].round(0)
        df_prov_n[num_cols] = df_prov_n[num_cols].astype(int)
        num_cols = df_prov_n.select_dtypes(include=['float64', 'int']).columns
        df_prov_n[num_cols] = df_prov_n[num_cols].applymap(lambda x: '{:,.0f}'.format(x).replace(',', '#').replace('.', ',').replace('#', '.'))
        return(df_prov_n)
@st.cache_resource
def tipog_donut(df_hab, prov):
    donut_tipog = df_hab[df_hab["PROVINCIA"]==prov][["PROVINCIA", "TIPO"]].value_counts(normalize=True).reset_index()
    donut_tipog.columns = ["PROVINCIA", "TIPO", "Habitatges en oferta"]
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=donut_tipog["TIPO"],
        values=donut_tipog["Habitatges en oferta"],
        hole=0.5, 
        showlegend=True, 
        marker=dict(
            colors=["#008B6C", "#00D0A3",  "#66b9a7", "#DAE4E0"], 
            line=dict(color='#FFFFFF', width=1) 
        ),
        textposition='outside',
        textinfo='percent+label' 
    ))
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    fig.update_layout(
        title=f'Habitatges en oferta per tipologia',
        font=dict(size=12),
        legend=dict(
            x=0.85,  # Set legend position
            y=0.85
        )
    )
    return(fig)
@st.cache_resource
def num_dorms_prov(df_hab, prov):
    table33_prov =  pd.crosstab(df_hab["PROVINCIA"], df_hab["Total dormitoris"]).reset_index().rename(columns={"PROVINCIA":"Província"})
    table33_prov = table33_prov[table33_prov["Província"]==prov].drop("Província", axis=1).T.reset_index()
    table33_prov.columns = ["Total dormitoris", "Habitatges en oferta"]

    fig = go.Figure(go.Bar(x=table33_prov["Total dormitoris"], y=table33_prov["Habitatges en oferta"], marker_color='#66b9a7'))
    fig.layout.yaxis = dict(title="Habitages en oferta", tickformat=",d")
    fig.update_layout(
        title=f"Habitatges en oferta segons nombre d'habitacions",
        xaxis_title="Nombre d'habitacions",
    )
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return(fig)
@st.cache_resource
def qualitats_prov(df_hab, prov):
    table62_hab = df_hab[df_hab["PROVINCIA"]==prov][["Aire condicionat","Bomba de calor","Aerotèrmia","Calefacció","Preinstal·lació d'A.C./B. Calor/Calefacció",'Parquet','Armaris encastats','Placa de cocció amb gas','Placa de cocció vitroceràmica',"Placa d'inducció",'Plaques solars']].sum(axis=0)
    table62_hab = pd.DataFrame({"Equipaments":table62_hab.index, "Total":table62_hab.values})
    table62_hab = table62_hab.set_index("Equipaments").apply(lambda row: (row / df_hab.shape[0])*100)
    table62_hab = table62_hab.sort_values("Total", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=table62_hab["Total"],  # Use values as x-axis data
        y=table62_hab.index,  # Use categories as y-axis data
        orientation="h",  # Set orientation to horizontal
        marker=dict(color="#66b9a7"),  # Set bar color
    ))
    fig.update_layout(
        title="Qualitats d'habitatges en oferta",
        xaxis_title="% d'habitatges en oferta",
        yaxis_title="Qualitats",
    )
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return(fig)
@st.cache_resource
def equipaments_prov(df_hab, prov):
    table67_hab = df_hab[df_hab["PROVINCIA"]==prov][["Zona enjardinada", "Parc infantil", "Piscina comunitària", "Traster", "Ascensor", "Equipament Esportiu", "Sala de jocs", "Sauna", "Altres", "Cap dels anteriors"]].sum(axis=0)
    table67_hab = pd.DataFrame({"Equipaments":table67_hab.index, "Total":table67_hab.values})
    table67_hab = table67_hab.set_index("Equipaments").apply(lambda row: row.mul(100) / df_hab.shape[0])
    table67_hab = table67_hab.sort_values("Total", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=table67_hab["Total"],  # Use values as x-axis data
        y=table67_hab.index,  # Use categories as y-axis data
        orientation="h",  # Set orientation to horizontal
        marker=dict(color="#66b9a7"),  # Set bar color
    ))
    fig.update_layout(
        title="Equipaments d'habitatges en oferta",
        xaxis_title="% d'habitatges en oferta",
        yaxis_title="Equipaments",
    )
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return(fig)
@st.cache_resource
def tipo_obra_prov(df_hab, prov):
    table38hab_prov = df_hab[["PROVINCIA", "TIPH"]].value_counts().reset_index().sort_values(["PROVINCIA", "TIPH"])
    table38hab_prov.columns = ["PROVINCIA", "TIPOLOGIA", "Habitatges"]
    table38hab_prov = table38hab_prov.pivot_table(index="PROVINCIA", columns="TIPOLOGIA", values="Habitatges").reset_index().rename(columns={"PROVINCIA":"Província"})
    table38hab_prov = table38hab_prov[table38hab_prov["Província"]==prov].drop("Província", axis=1).T.reset_index()
    table38hab_prov.columns = ["Tipus", "Habitatges en oferta"]
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=table38hab_prov["Tipus"],
        values=table38hab_prov["Habitatges en oferta"],
        hole=0.5, 
        showlegend=True, 
        marker=dict(
            colors=["#008B6C",  "#00D0A3"], 
            line=dict(color='#FFFFFF', width=1) 
        ),
        textposition='outside',
        textinfo='percent+label' 
    ))
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    fig.update_layout(
        title=f'Habitatges en oferta per tipus (obra nova o rehabilitació)',
        font=dict(size=12),
        legend=dict(
            x=0.7,  # Set legend position
            y=0.85
        )
    )
    return(fig)
@st.cache_resource
def cons_acabats(df_prom, df_hab, prov):
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=["Habitatges en construcció", "Habitatges acabats"],
        values=[metric_estat(df_prom, df_hab, prov)[0] - metric_estat(df_prom, df_hab, prov)[1], metric_estat(df_prom, df_hab, prov)[1]],
        hole=0.5, 
        showlegend=True, 
        marker=dict(
            colors=["#008B6C",  "#00D0A3"], 
            line=dict(color='#FFFFFF', width=1) 
        ),
        textposition='outside',
        textinfo='percent+label' 
    ))
    fig.update_layout(
        title=f'Habitatges en construcció i acabats',
        font=dict(size=12),
        legend=dict(
            x=0.7,  # Set legend position
            y=1.1
        )
    )
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return(fig)
@st.cache_resource
def metric_estat(df_prom, df_hab, prov):
    table11_prov = df_prom[["PROVINCIA", "HABIP"]].groupby("PROVINCIA").sum().reset_index()
    hab_oferta = table11_prov[table11_prov["PROVINCIA"]==prov].iloc[0,1]
    table17_hab_prov = df_hab[["PROVINCIA", "ESTO"]].value_counts().reset_index().sort_values(["PROVINCIA", "ESTO"])
    table17_hab_prov.columns = ["PROVINCIA","ESTAT", "PROMOCIONS"]
    table17_hab_prov = table17_hab_prov.pivot_table(index="PROVINCIA", columns="ESTAT", values="PROMOCIONS").reset_index()
    table17_hab_prov = table17_hab_prov[["PROVINCIA","Claus en mà"]].rename(columns={"PROVINCIA": "Província","Claus en mà":"Acabats sobre habitatges en oferta"})
    acabats_oferta = table17_hab_prov[table17_hab_prov["Província"]==prov].iloc[0,1]
    return([hab_oferta, acabats_oferta])
@st.cache_resource
def metric_rehab(df_hab, prov):
    table38hab_prov = df_hab[["PROVINCIA", "TIPH"]].value_counts().reset_index().sort_values(["PROVINCIA", "TIPH"])
    table38hab_prov.columns = ["PROVINCIA", "TIPOLOGIA", "Habitatges"]
    table38hab_prov = table38hab_prov.pivot_table(index="PROVINCIA", columns="TIPOLOGIA", values="Habitatges").reset_index().rename(columns={"PROVINCIA":"Província"})
    table38hab_prov = table38hab_prov[table38hab_prov["Província"]==prov].drop("Província", axis=1).T.reset_index()
    table38hab_prov.columns = ["Tipus", "Habitatges en oferta"]
    return([table38hab_prov.iloc[0,1], table38hab_prov.iloc[1,1]]) 
############################################################# MUNICIPIS FUNCIONS #############################################
@st.cache_resource
def carregant_dades():
    with open(path + 'DT_oferta.json', 'r') as outfile:
        list_of_df = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    bbdd_estudi_prom = list_of_df[0].copy()
    bbdd_estudi_hab = list_of_df[1].copy()
    bbdd_estudi_prom_2023 = list_of_df[2].copy()
    bbdd_estudi_hab_2023 = list_of_df[3].copy()
    mun_2018_2019 = list_of_df[4].copy()
    mun_2020_2021 = list_of_df[5].copy()
    mun_2022 = list_of_df[6].copy()
    mun_2023 = list_of_df[7].copy()
    maestro_estudi = list_of_df[8].copy()
    dis_2018_2019 = list_of_df[9].copy()
    dis_2020_2021 = list_of_df[10].copy()
    dis_2022 = list_of_df[11].copy()
    dis_2023 = list_of_df[12].copy()
    table117_22 = list_of_df[13].copy()
    table121_22 = list_of_df[14].copy()
    table125_22 = list_of_df[15].copy()
    table117_23 = list_of_df[16].copy()
    table121_23 = list_of_df[17].copy()
    table125_23 = list_of_df[18].copy()
    return([bbdd_estudi_prom, bbdd_estudi_hab, bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, mun_2018_2019, mun_2020_2021,
            mun_2022, mun_2023, maestro_estudi, dis_2018_2019, dis_2020_2021, dis_2022, dis_2023, table117_22,
            table121_22, table125_22, table117_23, table121_23, table125_23])


bbdd_estudi_prom, bbdd_estudi_hab, bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, \
mun_2018_2019, mun_2020_2021, mun_2022, mun_2023, maestro_estudi, dis_2018_2019, \
dis_2020_2021, dis_2022, dis_2023, table117_22, table121_22, table125_22, table117_23, \
table121_23, table125_23 = carregant_dades()            

@st.cache_resource
def data_text_mun(df_hab, df_hab_mod, selected_mun):
    table80_mun = df_hab_mod[df_hab_mod["Municipi"]==selected_mun][["Municipi", "TIPOG", "Superfície útil", "Preu mitjà", "Preu m2 útil"]].groupby(["Municipi"]).agg({"Municipi":['count'], "Superfície útil": [np.mean], "Preu mitjà": [np.mean], "Preu m2 útil": [np.mean]}).reset_index()
    table25_mun = df_hab[df_hab["Municipi"]==selected_mun][["Municipi", "TIPOG"]].value_counts(normalize=True).reset_index().rename(columns={0:"Proporció"})
    table61_hab = df_hab[df_hab["Municipi"]==selected_mun].groupby(['Total dormitoris']).size().reset_index(name='Proporcions').sort_values(by="Proporcions", ascending=False)
    table61_lav = df_hab[df_hab["Municipi"]==selected_mun].groupby(['Banys i lavabos']).size().reset_index(name='Proporcions').sort_values(by="Proporcions", ascending=False)

    try:
        proporcio_tipo = round(table25_mun[table25_mun["TIPOG"]=="Habitatges plurifamiliars"]["Proporció"].values[0]*100,2)
    except IndexError:
        proporcio_tipo = 0

    return([round(table80_mun["Preu mitjà"].values[0][0],2), round(table80_mun["Superfície útil"].values[0][0],2), 
            round(table80_mun["Preu m2 útil"].values[0][0],2), proporcio_tipo, 
            table61_hab["Total dormitoris"].values[0], table61_lav["Banys i lavabos"].values[0]])
@st.cache_resource
def plotmun_streamlit(data, selected_mun, kpi):
    df = data[(data['Municipi']==selected_mun)]
    fig = px.histogram(df, x=kpi, title= "", labels={'x':kpi, 'y':'Freqüència'})
    fig.data[0].marker.color = "#66b9a7"
    fig.layout.xaxis.title.text = kpi
    fig.layout.yaxis.title.text = 'Freqüència'
    mean_val = df[kpi].mean()
    fig.layout.shapes = [dict(type='line', x0=mean_val, y0=0, x1=mean_val, y1=1, yref='paper', xref='x', 
                            line=dict(color="black", width=2, dash='dot'))]
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    fig.layout.xaxis = dict(title=kpi, tickformat=",d")
    return(fig)
@st.cache_resource
def count_plot_mun(data, selected_mun):
    df = data[data['Municipi']==selected_mun]
    df = df["TIPOG"].value_counts().sort_values(ascending=True)
    fig = px.bar(df, y=df.index, x=df.values, orientation='h', title="", 
                labels={'x':"Número d'habitatges", 'y':"TIPOG"}, text= df.values)
    fig.layout.xaxis = dict(title="Nombre d'habitatges", tickformat=",d")
    fig.layout.yaxis.title.text = "Tipologia"
    fig.update_traces(marker=dict(color="#66b9a7"))
    max_width = 0.1
    fig.update_layout(bargap=(1 - max_width) / 2, bargroupgap=0)
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return fig
@st.cache_resource
def dormscount_plot_mun(data, selected_mun):
    df = data[data['Municipi']==selected_mun]
    custom_order = ["0D", "1D", "2D", "3D", "4D", "5+D"]
    df = df["Total dormitoris"].value_counts().reindex(custom_order)
    fig = px.bar(df,  y=df.values, x=df.index,title="", labels={'x':"Número d'habitacions", 'y':"Número d'habitatges"}, text= df.values)
    fig.layout.yaxis = dict(title="Nombre d'habitatges", tickformat=",d")
    fig.layout.xaxis.title.text = "Nombre d'habitacions"
    fig.update_traces(marker=dict(color="#66b9a7"))
    max_width = 0.1
    fig.update_layout(bargap=(1 - max_width) / 2, bargroupgap=0)
    for trace in fig.data:
        trace.text = [f"{val:,.0f}" if not np.isnan(val) else '' for val in trace.y]
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return fig
@st.cache_resource
def lavcount_plot_mun(data, selected_mun):
    df = data[data['Municipi']==selected_mun]

    df = df["Banys i lavabos"].value_counts().sort_values(ascending=True)
    fig = px.bar(df,  y=df.values, x=df.index,title="", labels={'x':"Número de lavabos", 'y':"Número d'habitatges"}, text= df.values)
    fig.layout.yaxis = dict(title="Nombre d'habitatges", tickformat=",d")
    fig.layout.xaxis.title.text = "Nombre de lavabos"
    fig.update_traces(marker=dict(color="#66b9a7"))
    max_width = 0.00001
    fig.update_layout(bargap=(1 - max_width) / 2, bargroupgap=0)
    for trace in fig.data:
        trace.text = [f"{val:,.0f}" if not np.isnan(val) else '' for val in trace.y]
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return fig
@st.cache_resource
def table_mun(Municipi, any_ini, any_fin):
    df_mun_filtered = df_final[(df_final["GEO"]==Municipi) & (df_final["Any"]>=any_ini) & (df_final["Any"]<=any_fin)].drop(["Àmbits territorials","Corones","Comarques","Província", "codiine"], axis=1).pivot(index=["Any"], columns=["Tipologia", "Variable"], values="Valor")
    df_mun_unitats = df_final[(df_final["GEO"]==Municipi) & (df_final["Any"]>=any_ini) & (df_final["Any"]<=any_fin)].drop(["Àmbits territorials","Corones","Comarques","Província", "codiine"], axis=1).drop_duplicates(["Any","Tipologia","Unitats"]).pivot(index=["Any"], columns=["Tipologia"], values="Unitats")
    df_mun_unitats.columns= [("HABITATGES PLURIFAMILIARS", "Unitats"), ("HABITATGES UNIFAMILIARS", "Unitats"), ("TOTAL HABITATGES", "Unitats")]
    df_mun_n = pd.concat([df_mun_filtered, df_mun_unitats], axis=1)
    # df_mun_n[("HABITATGES PLURIFAMILIARS", "Unitats %")] = (df_mun_n[("HABITATGES PLURIFAMILIARS", "Unitats")]/df_mun_n[("TOTAL HABITATGES", "Unitats")])*100
    # df_mun_n[("HABITATGES UNIFAMILIARS", "Unitats %")] = (df_mun_n[("HABITATGES UNIFAMILIARS", "Unitats")] /df_mun_n[("TOTAL HABITATGES", "Unitats")])*100
    df_mun_n = df_mun_n.sort_index(axis=1, level=[0,1])
    num_cols = df_mun_n.select_dtypes(include=['float64', 'Int64']).columns
    df_mun_n[num_cols] = df_mun_n[num_cols].round(0)
    df_mun_n[num_cols] = df_mun_n[num_cols].astype("Int64")
    num_cols = df_mun_n.select_dtypes(include=['float64', 'Int64']).columns
    df_mun_n[num_cols] = df_mun_n[num_cols].applymap(lambda x: '{:,.0f}'.format(x).replace(',', '#').replace('.', ',').replace('#', '.'))
    return(df_mun_n)
@st.cache_resource
def plot_mun_hist_units(selected_mun, variable_int, any_ini, any_fin):
    df_preus = df_vf_aux[(df_vf_aux['Variable']==variable_int) & (df_vf_aux['GEO']==selected_mun) & (df_vf_aux["Any"]>=any_ini) & (df_vf_aux["Any"]<=any_fin)].drop(['Variable'], axis=1).reset_index().drop('index', axis=1)
    df_preus['Valor'] = np.where(df_preus['Valor']==0, np.NaN, round(df_preus['Valor'], 1))
    df_preus['Any'] = df_preus['Any'].astype(int)
    df_preus = df_preus[df_preus["Tipologia"]!="TOTAL HABITATGES"]
    fig = px.bar(df_preus, x='Any', y='Valor', color='Tipologia', color_discrete_sequence=["#AAC4BA","#00D0A3"], range_y=[0, None], labels={'Valor': variable_int, 'Any': 'Any'}, text= "Valor")
    fig.layout.yaxis = dict(title= variable_int,tickformat=",d")
    valid_years = sorted(df_preus['Any'].unique())
    fig.update_xaxes(tickvals=valid_years)
    for trace in fig.data:
        trace.text = [f"{val:,.0f}" if not np.isnan(val) else '' for val in trace.y]
    max_width = 0.2
    fig.update_layout(bargap=(1 - max_width) / 2, bargroupgap=0)
    fig.update_layout(font=dict(size=13), legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=0.75))
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return fig
@st.cache_resource
def plot_mun_hist(selected_mun, variable_int, any_ini, any_fin):
    df_preus = df_vf[(df_vf['Variable']==variable_int) & (df_vf['GEO']==selected_mun) & (df_vf["Any"]>=any_ini) & (df_vf["Any"]<=any_fin)].drop(['Variable'], axis=1).reset_index().drop('index', axis=1)
    df_preus['Valor'] = np.where(df_preus['Valor']==0, np.NaN, round(df_preus['Valor'], 1))
    df_preus['Any'] = df_preus['Any'].astype(int)
    fig = px.bar(df_preus, x='Any', y='Valor', color='Tipologia', color_discrete_sequence=["#008B6C","#AAC4BA","#00D0A3"], range_y=[0, None], labels={'Valor': variable_int, 'Any': 'Any'}, text='Valor', barmode='group')
    fig.layout.yaxis = dict(title= variable_int,tickformat=",d")
    for trace in fig.data:
        trace.text = [f"{val:,.0f}" if not np.isnan(val) else '' for val in trace.y]
    max_width = 0.5
    fig.update_layout(bargap=(1 - max_width) / 2, bargroupgap=0)
    fig.update_layout(font=dict(size=13), legend=dict(orientation='h', yanchor='bottom', y=1, xanchor='right', x=0.75))
    fig.layout.paper_bgcolor = "#cce8e2"
    fig.layout.plot_bgcolor = "#cce8e2"
    return fig

######################################### IMPORTAMOS BBDD 2023 #########################################

@st.cache_resource
def tidy_bbdd_semestral(df_prom, df_hab, any):
    # bbdd_estudi_prom = pd.read_excel(path + 'P3007 BBDD desembre APCE.xlsx', sheet_name='Promocions 2023')
    bbdd_estudi_prom = df_prom.copy()
    bbdd_estudi_prom.columns = bbdd_estudi_prom.iloc[0,:]
    bbdd_estudi_prom = bbdd_estudi_prom[bbdd_estudi_prom["ESTUDI"]==any]
    bbdd_estudi_prom['TIPO_aux'] = np.where(bbdd_estudi_prom['TIPO'].isin([1,2]), 'Habitatges unifamiliars', 'Habitatges plurifamiliars')

    mapping = {1: 'Unifamiliars aïllats', 
            2: 'Unifamiliars adossats', 
            3: 'Plurifamiliars en bloc obert', 
            4: 'Plurifamiliars en bloc tancat'}

    mapping1 = {1: "De nova Construcció",
                2: "Rehabilitació integral"}

    mapping2 = {1: "Pendent d'enderroc", 
            2: "Solar", 
            3: "Buidat", 
            4: "Cimentació",
            5: "Estructura",
            6: "Tancaments exteriors",
            7: "Tancaments interiors",
            8: "Claus en mà",
            9: "NS/NC"}

    mapping3 = {
                    1: 'A',
                    1.2:"A",
                    2: 'B',
                    2.3: "B",
                    3: 'C',
                    4: 'D',
                    4.5: "D",
                    5: 'E',
                    5.3 : "C",
                    6: "F",
                    7: "G",
                    8: "En tràmits",
                    9: "Sense informació"
    }

    mapping4 = {
                    0: "Altres",
                    1: "Plaça d'aparcament opcional",
                    2: "Plaça d'aparcament inclosa",
                    3: "Sense plaça d'aparcament",
    }



    bbdd_estudi_prom['TIPO'] = bbdd_estudi_prom['TIPO'].map(mapping)

    bbdd_estudi_prom['TIPH'] = bbdd_estudi_prom['TIPH'].map(mapping1)


    bbdd_estudi_prom['ESTO'] = bbdd_estudi_prom['ESTO'].map(mapping2)

    bbdd_estudi_prom['QENERGC'] = bbdd_estudi_prom['QENERGC'].map(mapping3)

    bbdd_estudi_prom['APAR'] = bbdd_estudi_prom['APAR'].map(mapping4)


    # Importar BBDD habitatges
    # bbdd_estudi_hab = pd.read_excel(path + 'P3007 BBDD desembre APCE.xlsx', sheet_name='Habitatges 2023')
    bbdd_estudi_hab = df_hab.copy()
    bbdd_estudi_hab.columns = bbdd_estudi_hab.iloc[0,:]
    bbdd_estudi_hab = bbdd_estudi_hab[bbdd_estudi_hab["ESTUDI"]==any]





    # ["Total dormitoris","Banys i lavabos","Cuines estàndard","Cuines americanes","Terrasses, balcons i patis","Estudi/golfes","Safareig","Altres interiors","Altres exteriors"]

    # ["DORM", "LAV", "cuina_normal", "cuina_amer", "TER", "Golfes", "Safareig","Altres interiors","Altres exteriors" ]

    bbdd_estudi_hab['TIPOG'] = np.where(bbdd_estudi_hab['TIPO'].isin([1,2]), 'Habitatges unifamiliars', 'Habitatges plurifamiliars')
    bbdd_estudi_hab['TIPO'] = bbdd_estudi_hab['TIPO'].map(mapping)
    bbdd_estudi_hab['QENERGC'] = bbdd_estudi_hab['QENERGC'].map(mapping3)
    bbdd_estudi_hab['APAR'] = bbdd_estudi_hab['APAR'].map(mapping4)

    bbdd_estudi_hab = bbdd_estudi_hab.dropna(axis=1 , how ='all')



    bbdd_estudi_hab = bbdd_estudi_hab.rename(columns = {'V0006':'Total dormitoris_aux', 
                                                            "DORM": "Total dormitoris",
                                                            "LAV": "Banys i lavabos",
                                                            "TER": "Terrasses, balcons i patis",
                                                            'NOMD01C':'Superfície útil',
                                                            "Preu_m2_util": "Preu m2 útil",
                                                            "NOMD01F_2022": "Preu mitjà",
                                                            'NOMD01P':'Estudi/golfes', 
                                                            'NOMD01Q':'Safareig', 
                                                            'NOMD01K': 'Cuines estàndard', 
                                                            'NOMD01L': 'Cuines americanes', 
                                                            "NOMD01R": "Altres interiors", 
                                                            "NOMD01S":"Altres exteriors"})

    bbdd_estudi_prom = bbdd_estudi_prom.rename(columns = {'V0006':'Total dormitoris_aux', 
                                                            "DORM": "Total dormitoris",
                                                            "LAV": "Banys i lavabos",
                                                            "TER": "Terrasses, balcons i patis",
                                                            'NOMD01C':'Superfície útil',
                                                            "Preu_m2_util": "Preu m2 útil",
                                                            'NOMD01P':'Estudi/golfes', 
                                                            'NOMD01Q':'Safareig', 
                                                            'NOMD01K': 'Cuines estàndard', 
                                                            'NOMD01L': 'Cuines americanes', 
                                                            "NOMD01R": "Altres interiors", 
                                                            "NOMD01S":"Altres exteriors"})


    # Canviar de nom tots els equipaments
    bbdd_estudi_hab = bbdd_estudi_hab.rename(columns = {'EQUIC_1': 'Zona enjardinada', 
                                                        'EQUIC_2': 'Parc infantil',
                                                        'EQUIC_3': 'Piscina comunitària', 
                                                        'EQUIC_4': 'Traster', 
                                                        'EQUIC_5': 'Ascensor', 
                                                        'EQUIC_6': 'Equipament Esportiu',  
                                                        'EQUIC_7': 'Sala de jocs', 
                                                        'EQUIC_8': 'Sauna', 
                                                        "EQUIC_9_50": "Altres",
                                                        'EQUIC_99': 'Cap dels anteriors'})
    bbdd_estudi_prom = bbdd_estudi_prom.rename(columns = {'EQUIC_1': 'Zona enjardinada', 
                                                        'EQUIC_2': 'Parc infantil',
                                                        'EQUIC_3': 'Piscina comunitària', 
                                                        'EQUIC_4': 'Traster', 
                                                        'EQUIC_5': 'Ascensor', 
                                                        'EQUIC_6': 'Equipament Esportiu',  
                                                        'EQUIC_7': 'Sala de jocs', 
                                                        'EQUIC_8': 'Sauna', 
                                                        "QUAL_ALTRES": "Altres",
                                                        'EQUIC_99': 'Cap dels anteriors'})
    bbdd_estudi_prom["Ascensor"] = np.where(bbdd_estudi_prom["Ascensor"]>=1, 1, bbdd_estudi_prom["Ascensor"])
    bbdd_estudi_hab["Ascensor"] = np.where(bbdd_estudi_hab["Ascensor"]>=1, 1, bbdd_estudi_hab["Ascensor"])


    # Canviar de nom totes les qualitats
    bbdd_estudi_hab = bbdd_estudi_hab.rename(columns = {"QUALIC_5": "Aire condicionat", 
                                                        "QUALIC_6": "Bomba de calor", 
                                                        "QUALI_A": "Aerotèrmia", 
                                                        'QUALIC_7':"Calefacció", 
                                                        'QUALIC_8':"Preinstal·lació d'A.C./B. Calor/Calefacció", 
                                                        'QUALIC_9': 'Parquet', 
                                                        'QUALIC_10':'Armaris encastats',
                                                        'QUALIC_12':'Placa de cocció amb gas',
                                                        'QUALIC_13':'Placa de cocció vitroceràmica',
                                                        "QUALIC_14":"Placa d'inducció",
                                                        'QUALIC_22':'Plaques solars'})


    bbdd_estudi_prom = bbdd_estudi_prom.rename(columns = {"QUALIC_5": "Aire condicionat", 
                                                        "QUALIC_6": "Bomba de calor", 
                                                        "QUALI_A": "Aerotèrmia", 
                                                        'QUALIC_7':"Calefacció", 
                                                        'QUALIC_8':"Preinstal·lació d'A.C./B. Calor/Calefacció", 
                                                        'QUALIC_9': 'Parquet', 
                                                        'QUALIC_10':'Armaris encastats',
                                                        'QUALIC_12':'Placa de cocció amb gas',
                                                        'QUALIC_13':'Placa de cocció vitroceràmica',
                                                        "QUALIC_14":"Placa d'inducció",
                                                        'QUALIC_22':'Plaques solars'})
    #  Canviar nom a tipus de calefacció
    bbdd_estudi_prom = bbdd_estudi_prom.rename(columns = {'CALEFC_3': 'De gasoil', 
                                                        'CALEFC_4': 'De gas natural', 
                                                        'CALEFC_5': 'De propà', 
                                                        'CALEFC_6': "D'electricitat", 
                                                        'CALEFC_9': "No s'indica tipus"})




    bbdd_estudi_prom['TIPV'] = np.where(bbdd_estudi_prom['TIPV_1'] >= 1, "Venda a través d'immobiliària independent",
                                        np.where(bbdd_estudi_prom['TIPV_2'] >= 1, "Venda a través d'immobiliaria del mateix promotor",
                                                np.where(bbdd_estudi_prom['TIPV_3'] >= 1, "Venda directa del promotor", "Sense informació")))


    bbdd_estudi_prom['TIPOL_VENDA'] = np.where(bbdd_estudi_prom['TIPOL_VENDA_1'] == 1, "0D",
                                        np.where(bbdd_estudi_prom['TIPOL_VENDA_2'] == 1, "1D",
                                                np.where(bbdd_estudi_prom['TIPOL_VENDA_3'] == 1, "2D",
                                                        np.where(bbdd_estudi_prom['TIPOL_VENDA_4'] == 1, "3D",
                                                            np.where(bbdd_estudi_prom['TIPOL_VENDA_5'] == 1, "4D", 
                                                                np.where(bbdd_estudi_prom['TIPOL_VENDA_6'] == 1, "5+D", "NA"))))))

                        
                                                    
    #  "Venda a través d'immobiliària independent", "Venda a través d'immobiliaria del mateix promotor", "Venda directa del promotor"

    bbdd_estudi_hab['TIPH'] = bbdd_estudi_hab['TIPH'].map(mapping1)

    bbdd_estudi_hab['ESTO'] = bbdd_estudi_hab['ESTO'].map(mapping2)


    vars = ['Zona enjardinada', 'Parc infantil', 'Piscina comunitària', 
            'Traster', 'Ascensor', 'Equipament Esportiu', 'Sala de jocs', 
            'Sauna', 'Altres', "Aire condicionat", "Bomba de calor", 
            "Aerotèrmia", "Calefacció", "Preinstal·lació d'A.C./B. Calor/Calefacció", 
            "Parquet", "Armaris encastats", 'Placa de cocció amb gas', 
            'Placa de cocció vitroceràmica', "Placa d'inducció", 'Plaques solars', "APAR"]
    vars_aux = ['Zona enjardinada', 'Parc infantil', 'Piscina comunitària', 
            'Traster', 'Ascensor', 'Equipament Esportiu', 'Sala de jocs', 
            'Sauna', 'Altres', "Aire condicionat", "Bomba de calor", 
            "Aerotèrmia", "Calefacció", "Preinstal·lació d'A.C./B. Calor/Calefacció", 
            "Parquet", "Armaris encastats", 'Placa de cocció amb gas', 
            'Placa de cocció vitroceràmica', "Placa d'inducció", 'Plaques solars', "Safareig","Terrasses, balcons i patis"]
    for i in vars:
        bbdd_estudi_prom[i] = bbdd_estudi_prom[i].replace(np.nan, 0)
    for i in vars_aux:
        bbdd_estudi_hab[i] = bbdd_estudi_hab[i].replace(np.nan, 0)
    bbdd_estudi_hab["Calefacció"] = bbdd_estudi_hab["Calefacció"].replace(' ', 0) 
    bbdd_estudi_prom["Calefacció"] = bbdd_estudi_prom["Calefacció"].replace(' ', 0) 


    bbdd_estudi_hab["Tram_Sup_util"] = bbdd_estudi_hab["Tram_Sup_util"].str.replace(" ", "")
    bbdd_estudi_hab["Tram_Sup_util"] = bbdd_estudi_hab["Tram_Sup_util"].str[3:]



    # Afegir categories a algunes columnes de la base de dades d'habitatge

    room_dict =  {i: f"{i}D" if i <= 4 else "5+D" for i in range(0, 20)}
    toilet_dict = {i: f"{i} Bany" if i <= 1 else "2 i més Banys" for i in range(1, 20)}
    bbdd_estudi_hab_mod = bbdd_estudi_hab.copy()

    bbdd_estudi_hab_mod['Total dormitoris'] = bbdd_estudi_hab_mod['Total dormitoris'].map(room_dict)
    bbdd_estudi_hab_mod['Banys i lavabos'] = bbdd_estudi_hab_mod['Banys i lavabos'].map(toilet_dict)
    bbdd_estudi_hab_mod["Terrasses, balcons i patis"] = np.where(bbdd_estudi_hab_mod["Terrasses, balcons i patis"]>=1, 1, 0)

    bbdd_estudi_hab["Nom DIST"] = bbdd_estudi_hab["Nom DIST"].str.replace(r'^\d{2}\s', '', regex=True)
    bbdd_estudi_hab_mod["Nom DIST"] = bbdd_estudi_hab_mod["Nom DIST"].str.replace(r'^\d{2}\s', '', regex=True)

    return([bbdd_estudi_prom, bbdd_estudi_hab, bbdd_estudi_hab_mod])

bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023 = tidy_bbdd_semestral(bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, 2023)
############################################################  IMPORTAR HISTÓRICO DE MUNICIPIOS 2016 - 2022 ################################################
@st.cache_resource
def import_hist_mun(df_1819, df_2021, df_22, df_23, maestro_df):
    # mun_2018_2019 = pd.read_excel(path + "Resum 2018 - 2019.xlsx", sheet_name="Municipis 2018-2019")
    mun_2018_2019 = df_1819.copy()
    mun_2019 = mun_2018_2019.iloc[:,14:27]

    # mun_2020_2021 = pd.read_excel(path + "Resum 2020 - 2021.xlsx", sheet_name="Municipis")
    mun_2020_2021 = df_2021.copy()
    mun_2020 = mun_2020_2021.iloc[:,:13]
    mun_2020 = mun_2020.dropna(how ='all',axis=0)
    mun_2021 = mun_2020_2021.iloc[:,14:27]
    mun_2021 = mun_2021.dropna(how ='all',axis=0)

    # mun_2022 = pd.read_excel(path + "Resum 2022.xlsx", sheet_name="Municipis")
    mun_2022 = df_22.copy()
    mun_2022 = mun_2022.iloc[:,14:27]
    mun_2022 = mun_2022.dropna(how ='all',axis=0)

    # mun_2023 = pd.read_excel(path + "Resum 2023.xlsx", sheet_name="Municipis")
    mun_2023 = df_23.copy()
    mun_2023 = mun_2023.iloc[:,14:27]
    mun_2023 = mun_2023.dropna(how ='all',axis=0)

    # maestro_estudi = pd.read_excel(path + "Maestro estudi_oferta.xlsx", sheet_name="Maestro")
    maestro_estudi = maestro_df.copy()

    return([mun_2019, mun_2020, mun_2021, mun_2022, mun_2023, maestro_estudi])
mun_2019, mun_2020, mun_2021, mun_2022, mun_2023, maestro_estudi = import_hist_mun(mun_2018_2019, mun_2020_2021, mun_2022, mun_2023, maestro_estudi)
############################################################  IMPORTAR HISTÓRICO DE DISTRITOS DE BCN 2016 - 2023 ################################################
@st.cache_resource
def tidy_data(mun_year, year):
    df =mun_year.T
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,:].reset_index()
    df.columns.values[:3] = ['Any', 'Tipologia', "Variable"]
    df['Tipologia'] = df['Tipologia'].ffill()
    df['Any'] = year
    geo = df.columns[3:].values
    df_melted = pd.melt(df, id_vars=['Any', 'Tipologia', 'Variable'], value_vars=geo, value_name='Valor')
    df_melted.columns.values[3] = 'GEO'
    return(df_melted)

############################################################  CALCULOS PROVINCIAS, AMBITOS TERRITORIALES Y COMARCAS ################################################
def weighted_mean(data):
    weighted_sum = (data['Valor'] * data['Unitats']).sum()
    sum_peso = data['Unitats'].sum()
    # data["Valor"] = weighted_sum / sum_peso
    return weighted_sum / sum_peso
@st.cache_resource
def geo_mun():
    df_vf_aux = pd.DataFrame()

    for df_frame, year in zip(["mun_2019", "mun_2020", "mun_2021", "mun_2022", "mun_2023"], [2019, 2020, 2021, 2022, 2023]):
        df_vf_aux = pd.concat([df_vf_aux, tidy_data(eval(df_frame), year)], axis=0)


    df_vf_aux['Variable']= np.where(df_vf_aux['Variable']=="Preu de     venda per      m² útil (€)", "Preu de venda per m² útil (€)", df_vf_aux['Variable'])
    df_vf_aux['Valor'] = pd.to_numeric(df_vf_aux['Valor'], errors='coerce')
    df_vf_aux['GEO'] = np.where(df_vf_aux['GEO']=="Municipis de Catalunya", "Catalunya", df_vf_aux['GEO'])
    df_vf_aux = df_vf_aux[~df_vf_aux['GEO'].str.contains("província|Província|Municipis")]

    df_vf_merged = pd.merge(df_vf_aux, maestro_estudi, how="left", on="GEO")
    df_vf_merged = df_vf_merged[~df_vf_merged["Província"].isna()].dropna(axis=1, how="all")
    df_vf = df_vf_merged[df_vf_merged["Variable"]!="Unitats"]
    df_unitats = df_vf_merged[df_vf_merged["Variable"]=="Unitats"].drop("Variable", axis=1)
    df_unitats = df_unitats.rename(columns={"Valor": "Unitats"})
    # df_vf[df_vf["Província"].isna()]["GEO"].unique()
    df_final_cat = pd.merge(df_vf, df_unitats, how="left")
    df_final = df_final_cat[df_final_cat["GEO"]!="Catalunya"]
    df_final_cat_aux1 = df_final_cat[df_final_cat["GEO"]=="Catalunya"][["Any", "Tipologia", "Variable","Valor"]]
    cat_df_aux2_melted = pd.melt(df_final_cat[df_final_cat["GEO"]=="Catalunya"][["Any", "Tipologia", "Unitats"]], id_vars=["Any", "Tipologia"], var_name="Variable", value_name="Unitats")
    cat_df_aux2_melted["Unitats"] = cat_df_aux2_melted["Unitats"].astype("int64")
    cat_df_aux2_melted = cat_df_aux2_melted.rename(columns={"Unitats":"Valor"})
    df_final_cat = pd.concat([df_final_cat_aux1, cat_df_aux2_melted], axis=0)


    ambits_df_aux1 = df_final.groupby(["Any", "Tipologia", "Variable", "Àmbits territorials"]).apply(weighted_mean).reset_index().rename(columns= {0:"Valor"})
    ambits_df_aux2 = df_final[["Any","Àmbits territorials","Tipologia", "GEO", "Unitats"]].drop_duplicates(["Any","Àmbits territorials","Tipologia", "GEO", "Unitats"]).groupby(["Any", "Àmbits territorials", "Tipologia"]).sum().reset_index()
    ambits_df_aux2_melted = pd.melt(ambits_df_aux2, id_vars=["Any", "Tipologia", "Àmbits territorials"], var_name="Variable", value_name="Unitats")
    ambits_df_aux2_melted["Unitats"] = ambits_df_aux2_melted["Unitats"].astype("int64")
    ambits_df_aux2_melted = ambits_df_aux2_melted.rename(columns={"Unitats":"Valor"})
    ambits_df = pd.concat([ambits_df_aux1, ambits_df_aux2_melted], axis=0)
    ambits_df = ambits_df.rename(columns={"Àmbits territorials":"GEO"})

    comarques_df_aux1 = df_final.groupby(["Any", "Tipologia", "Variable", "Comarques"]).apply(weighted_mean).reset_index().rename(columns= {0:"Valor"}).dropna(axis=0)
    comarques_df_aux2 = df_final[["Any","Comarques","Tipologia", "GEO", "Unitats"]].drop_duplicates(["Any","Comarques","Tipologia", "GEO", "Unitats"]).groupby(["Any", "Comarques", "Tipologia"]).sum().reset_index()
    comarques_df_aux2_melted = pd.melt(comarques_df_aux2, id_vars=["Any", "Tipologia", "Comarques"], var_name="Variable", value_name="Unitats")
    comarques_df_aux2_melted["Unitats"] = comarques_df_aux2_melted["Unitats"].astype("int64")
    comarques_df_aux2_melted = comarques_df_aux2_melted.rename(columns={"Unitats":"Valor"})
    comarques_df = pd.concat([comarques_df_aux1, comarques_df_aux2_melted], axis=0)
    comarques_df = comarques_df.rename(columns={"Comarques":"GEO"})


    provincia_df_aux1 = df_final.groupby(["Any", "Tipologia", "Variable", "Província"]).apply(weighted_mean).reset_index().rename(columns= {0:"Valor"})
    provincia_df_aux2 = df_final[["Any","Província","Tipologia", "GEO", "Unitats"]].drop_duplicates(["Any","Província","Tipologia", "GEO", "Unitats"]).groupby(["Any", "Província", "Tipologia"]).sum().reset_index()
    provincia_df_aux2_melted = pd.melt(provincia_df_aux2, id_vars=["Any", "Tipologia", "Província"], var_name="Variable", value_name="Unitats")
    provincia_df_aux2_melted["Unitats"] = provincia_df_aux2_melted["Unitats"].astype("int64")
    provincia_df_aux2_melted = provincia_df_aux2_melted.rename(columns={"Unitats":"Valor"})
    provincia_df = pd.concat([provincia_df_aux1, provincia_df_aux2_melted], axis=0)
    provincia_df = provincia_df.rename(columns={"Província":"GEO"})

    return([df_vf_aux, df_vf, df_final_cat, df_final, ambits_df, comarques_df, provincia_df])

df_vf_aux, df_vf, df_final_cat, df_final, ambits_df, comarques_df, provincia_df = geo_mun()


def filedownload(df, filename):
    towrite = io.BytesIO()
    df.to_excel(towrite, encoding='latin-1', index=True, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode("latin-1")
    href = f"""<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">
    <button class="download-button">Descarregar</button></a>"""
    return href


################################################## ANALISIS DE MERCAT APCE #####################################################
if selected=="Análisis de mercado APCE":
    mun_names = sorted([name for name in df_vf[(df_vf["Any"]==2023) & (~df_vf["Valor"].isna())]["GEO"].unique() if name != "Catalunya"])
    left_col, right_col = st.columns((1, 1))
    with left_col:
        selected_geo = st.radio("**Àrea geogràfica**", ("Províncies", "Municipis"))
    with right_col:
        if selected_geo=="Municipis":
            selected_mun = st.selectbox('**Municipi seleccionat:**', mun_names, index= mun_names.index("Barcelona"))
        if selected_geo=="Províncies":
            prov_names = ["Barcelona", "Girona", "Tarragona", "Lleida"]
            selected_prov = st.selectbox('', prov_names, index= prov_names.index("Barcelona"))
    if selected_geo=="Províncies":
        st.subheader(f"PROVÍNCIA DE {selected_prov.upper()}")
        if selected_prov=="Barcelona":
            st.write(f"""Els municipis analitzats a l’Estudi de l’Oferta d’Habitatge
                        de Nova Construcció a Catalunya 2023 que pertanyen a
                        la província de Barcelona es situen de mitjana en primera
                        posició pel que fa al preu (385.036€)
                        i també quant al preu €/m\u00b2 de superfície útil (4.757€),
                        ambdós influenciats pel municipi de Barcelona en incidir
                        sobre les mitjanes de forma determinant, tant per la seva
                        aportació quantitativa com qualitativa. Addicionalment,
                        Barcelona se situa com la quarta província pel que fa a
                        la mitjana de superfície, de 79,6m\u00b2.
                        El número de promocions en oferta als municipis estudiats
                        a la província de Barcelona l’any 2023 va ser
                        de 894, amb 8.012 habitatges, xifra que representa el
                        79,6% del total de les promocions estudiades. El percentatge
                        d’habitatges que restaven per vendre és del
                        38,7% sobre un total de 20.725 habitatges existents
                        dins les promocions (en el moment d’elaborar aquest
                        estudi ja estaven venuts el 61,3% dels habitatges, majoritàriament
                        sobre plànol).
                        Pel que fa als habitatges unifamiliars, la mitjana de
                        superficie a la província se situa en 149,0 m\u00b2, la mitjana
                        de preu de venda en 593.401€ i la de preu m\u00b2 útil
                        en 3.942,8€. Pel que fa als habitatges plurifamiliars,
                        la superfície mitjana és de 76,8 m\u00b2, amb una mitjana
                        de preu de venda de 376.425€, i un preu mitjà per m\u00b2
                        útil de 4.790,2€.""")
        if selected_prov=="Girona":
            st.write(f"""Els municipis analitzats a l’Estudi de l'Oferta d'Habitatge de 2023 que pertanyen a
                        la província de Girona es situen de mitjana en tercera
                        posició respecte a la superfície útil (82,9 m\u00b2) i en segona
                        posició respecte al preu mitjà (349.647€) i el preu m\u00b2
                        de superfície útil (4.244,9€).
                        Pel que fa als habitatges plurifamiliars, la superfície
                        mitjana es situa en els 79,6 m\u00b2, amb un preu mitjà
                        de 343.984€, i un preu per m\u00b2 de superfície útil
                        de 4.295€. Respecte dels habitatges unifamiliars,
                        aquestes mitjanes són de 140,1 m\u00b2 de superfície, un
                        preu mitjà de 446.945€ i un preu per m\u00b2 de superfície
                        útil de 3.392€.
                        El nombre de promocions en oferta als municipis estudiats
                        de la província de Girona al 2023 era de 109 (un 9,7%
                        del total de les promocions estudiades a Catalunya), que contenen 1.200
                        habitatges en venda.
                        El percentatge d’habitatges que restaven per vendre és
                        del 51,3% sobre un total de 2.338 habitatges existents
                        a les promocions de la província.""")
        if selected_prov=="Tarragona":
            st.write(f"""Els municipis analitzats a l’Estudi de l'Oferta d'Habitatge de 2023 que pertanyen a la
                        província de Tarragona es situen de mitjana en segona
                        posició pel que fa a superfície (83,5m\u00b2), i
                        en tercera posició tant pel que fa a les mitjanes de preu
                        (254.667€) i de preu per m\u00b2 de superfície útil (3.029€).
                        Per tipologies d’habitatges, en els habitatges unifamiliars
                        les mitjanes registrades són: 140,5m\u00b2 de superfície,
                        amb un preu mitjà de 441.762, i un preu per m\u00b2 de superfície
                        útil de 3.354€. Pel que fa als habitatges plurifamiliars,
                        la superfície mitjana se situa en els 74,7m\u00b2, amb
                        un preu mitjà de 225.741€, i un preu per m\u00b2 útil de 2.979€.
                        El nombre de promocions en oferta als municipis estudiats
                        a la província de Tarragona al 2023 va ser de 84, xifra
                        que representa un 7,4% del total de les promocions
                        estudiades a Catalunya. El percentatge d’habitatges que restaven per vendre és
                        del 32,3% sobre un total de 2.175 habitatges existents
                        a les promocions de la província.""")
        if selected_prov=="Lleida":
            st.write(f"""Els municipis analitzats a l’Estudi de l'Oferta d'Habitatge de 2023 que pertanyen
                        a la província de Lleida obtenen la mitjana més alta de
                        superfície (86,6 m\u00b2) i es situen en quarta posició pel que
                        fa el preu del m\u00b2 de superfície útil (2.428€).
                        Quant als habitatges plurifamiliars, la superfície mitjana
                        provincial és de 82,7 m\u00b2 constituint un preu mitjà per m\u00b2 útil de
                        2.332€. En el cas dels habitatges unifamiliars,
                        aquestes quantitats són de 134,5 m\u00b2 de superfície mitjana
                        i de 3.586 la mitjana del preu m\u00b2 de superfície útil.
                        El nombre de promocions en oferta als municipis estudiats
                        a la província de Lleida al 2023 va ser de 36 (amb 382
                        habitatges en venda), dada que representa un 3,2% del
                        total de les promocions estudiades. El
                        percentatge d’habitatges que restaven per vendre és del
                        41,9% sobre un total de 911 habitatges existents a les
                        promocions a la província.""")
        st.markdown(table_geo(selected_prov, 2019, 2023, "Províncies").to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_geo(selected_prov, 2019, 2023, "Províncies"), f"Estudi_oferta_{selected_prov}.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(tipog_donut(bbdd_estudi_hab_2023, selected_prov), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(num_dorms_prov(bbdd_estudi_hab_mod_2023, selected_prov), use_container_width=True, responsive=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(qualitats_prov(bbdd_estudi_hab_2023, selected_prov), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(equipaments_prov(bbdd_estudi_hab_2023, selected_prov), use_container_width=True, responsive=True)
        left_col, right_col = st.columns((2, 1))
        with left_col:
            st.plotly_chart(cons_acabats(bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, selected_prov), use_container_width=True, responsive=True)
        with right_col:
            st.markdown("")
            st.markdown("")
            st.metric("**Habitatges en oferta**", format(int(metric_estat(bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, selected_prov)[0]), ",d"))
            st.metric("**Habitatges en construcció**", format(int(metric_estat(bbdd_estudi_prom_2023,  bbdd_estudi_hab_2023, selected_prov)[0] - metric_estat(bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, selected_prov)[1]), ",d"))
            st.metric("**Habitatges acabats**", format(int(metric_estat(bbdd_estudi_prom_2023, bbdd_estudi_hab_2023, selected_prov)[1]), ",d"))
        left_col, right_col = st.columns((2, 1))
        with left_col:
            st.plotly_chart(tipo_obra_prov(bbdd_estudi_hab_2023, selected_prov), use_container_width=True, responsive=True)
        with right_col:
            st.markdown("")
            st.markdown("")
            st.metric("**Habitatges de nova construcció**", format(int(metric_rehab(bbdd_estudi_hab_2023, selected_prov)[0]), ",d"))
            st.metric("**Habitatges de rehabilitació integral**", format(int(metric_rehab(bbdd_estudi_hab_2023, selected_prov)[1]), ",d"))     

    if selected_geo=="Municipis":
        st.subheader(f"MUNICIPI DE {selected_mun.upper().split(',')[0].strip()}")
        st.markdown(f"""Els resultats de l'Estudi d'Oferta de nova construcció del 2023 pel municipi de {selected_mun.split(',')[0].strip()} mostren que el preu mitjà dels habitatges en venda es troba 
        en {data_text_mun(bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023, selected_mun)[0]:,.1f} € amb una superfície mitjana útil de {data_text_mun(bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023, selected_mun)[1]:,.1f} m\u00b2. Per tant, el preu per m\u00b2 útil es troba en {data_text_mun(bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023, selected_mun)[2]:,.1f} € de mitjana. Per tipologies, els habitatges plurifamiliars
        representen el {data_text_mun(bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023, selected_mun)[3]:,.1f}% sobre el total d'habitatges, la resta corresponen a habitatges unifamiliars. L'habitatge modal o més freqüent de nova construcció té {data_text_mun(bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023, selected_mun)[4]} habitacions i {data_text_mun(bbdd_estudi_hab_2023, bbdd_estudi_hab_mod_2023, selected_mun)[5]} banys o lavabos.""")
        left_col, right_col = st.columns((1, 1))
        with left_col:
            st.markdown(f"""**Distribució de Preus per m\u00b2 útil**""")
            st.plotly_chart(plotmun_streamlit(bbdd_estudi_hab_mod_2023, selected_mun,"Preu m2 útil"), use_container_width=True, responsive=True)
        with right_col:
            st.markdown(f"""**Distribució de Superfície útil**""")
            st.plotly_chart(plotmun_streamlit(bbdd_estudi_hab_mod_2023, selected_mun, "Superfície útil"), use_container_width=True, responsive=True)
        st.markdown(f"""
        **Tipologia d'habitatges de les promocions**
        """)
        st.plotly_chart(count_plot_mun(bbdd_estudi_hab_mod_2023, selected_mun), use_container_width=True, responsive=True)
        left_col, right_col = st.columns((1, 1))
        with left_col:
            st.markdown("""**Habitatges a la venda segons número d'habitacions**""")
            st.plotly_chart(dormscount_plot_mun(bbdd_estudi_hab_mod_2023, selected_mun), use_container_width=True, responsive=True)

        with right_col:
            st.markdown("""**Habitatges a la venda segons número de Banys i lavabos**""")
            st.plotly_chart(lavcount_plot_mun(bbdd_estudi_hab_mod_2023, selected_mun), use_container_width=True, responsive=True)
        st.subheader("Comparativa amb anys anteriors: Municipi de " + selected_mun.split(',')[0].strip())
        st.markdown(table_mun(selected_mun, 2019, 2023).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_mun(selected_mun, 2019, 2023), f"Estudi_oferta_{selected_mun}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        left_col, right_col = st.columns((1, 1))
        with left_col:
            st.markdown("""**Evolució dels habitatges de nova construcció per tipologia d'habitatge**""")
            st.plotly_chart(plot_mun_hist_units(selected_mun, "Unitats", 2019, 2023), use_container_width=True, responsive=True)
        with right_col:
            st.markdown("""**Evolució de la superfície útil mitjana per tipologia d'habitatge**""")
            st.plotly_chart(plot_mun_hist(selected_mun, 'Superfície mitjana (m² útils)', 2019, 2023), use_container_width=True, responsive=True)
        left_col, right_col = st.columns((1, 1))
        with left_col:
            st.markdown("""**Evolució del preu de venda per m\u00b2 útil  per tipologia d'habitatge**""")
            st.plotly_chart(plot_mun_hist(selected_mun, "Preu de venda per m² útil (€)", 2019, 2023), use_container_width=True, responsive=True)
        with right_col:
            st.markdown("""**Evolució del preu venda mitjà per tipologia d'habitatge**""")
            st.plotly_chart(plot_mun_hist(selected_mun, "Preu mitjà de venda de l'habitatge (€)", 2019, 2023), use_container_width=True, responsive=True)