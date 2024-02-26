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
    options=["Análisis de mercado", "Edificabilidad","Análisis estático","Análisis dinámico", "Resumen de resultados"],  # Dropdown menu
    icons=[None, None, None, None],  # Icons for dropdown menu
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "0px important!", "background-color": "#edf1fc", "align":"center", "overflow":"hidden"},
        "icon": {"color": "#def3fa", "font-size": "17px"},
        "nav-link": {
            "font-size": "17px",
            "text-align": "center",
            "font-weight": "bold",
            "color":"black",
            "padding": "5px",
            "--hover-color": "#ADD8E6",
            "background-color": "#edf1fc",
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
    left, center, right= st.columns((1,1,1))
    with center:
        selected_propuesta = st.radio("", ("Propuesta 1", "Propuesta 2", "Propuesta 3", "Comparativa"), horizontal=True)
    if (selected_propuesta=="Propuesta 1") or (selected_propuesta=="Propuesta 2") or (selected_propuesta=="Propuesta 3"):
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing_data = conn.read(worksheet="Edificabilidad" + selected_propuesta[-1], usecols=list(range(3)), ttl=5)
        existing_data = existing_data.dropna(how="all")
        left, right = st.columns((1,1))
        for index, row in existing_data.iterrows():
            if index % 2 == 0:
                with left:
                    input_value = st.number_input(f"**{row['Elemento']}:**", value=row['Valor'])
                    exec(f"{row['Nombre']} = {input_value}")
            else:
                with right:
                    input_value = st.number_input(f"**{row['Elemento']}:**", value=row['Valor'])
                    exec(f"{row['Nombre']} = {input_value}")
        submit_button = st.button(label="Guardar proposta")
        result_df = pd.DataFrame(columns=['Nombre', 'Valor'])
        if submit_button:
            for variable_name in existing_data['Nombre']:
                variable_value = locals().get(variable_name, None)  # Get the value of the variable by name
                result_df = result_df.append({'Nombre': variable_name, 'Valor': variable_value}, ignore_index=True)
            result_df = pd.concat([existing_data.iloc[:,0], result_df], axis=1)
            conn.update(worksheet="Edificabilidad", data=result_df)
            st.success("¡Propuesta guardada!")
    if selected_propuesta=="Comparativa":
        num_propuesta_edifi = ["Edificabilidad1", "Edificabilidad2", "Edificabilidad3"]
        propuesta_edifi_df = []
        for i in num_propuesta_edifi:
            conn = st.connection("gsheets", type=GSheetsConnection)
            existing_data = conn.read(worksheet=i, usecols=list(range(3)), ttl=5)
            existing_data = existing_data.dropna(how="all")
            propuesta_edifi_df.append(existing_data)
        def bar_plotly(df, title_main, title_y):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["Elemento"],
                y=df["Valor"],
                name="m2",
                text=df["Valor"],  # Use the "Valor" column for labels
                textposition='auto'  # Position the labels automatically
            ))
            fig.update_layout(
                title=title_main,
                xaxis=dict(title=""),
                yaxis=dict(title=title_y, tickformat=",d"),
                paper_bgcolor="#edf1fc",
                plot_bgcolor='#edf1fc'
            )
            return fig
        left, center, right= st.columns((1,1,1))
        with left:
            propuesta1_edifi = propuesta_edifi_df[0]
            st.markdown(f'<div class="custom-box">PROPUESTA 1</div>', unsafe_allow_html=True)   
            st.metric(label="**Número de viviendas**", value=int(propuesta1_edifi[propuesta1_edifi["Nombre"]=="num_viviendas"]["Valor"]))
            st.metric(label="**Superficie de viviendas**", value=int(propuesta1_edifi[propuesta1_edifi["Nombre"]=="super_viviendas"]["Valor"]))
            st.metric(label="**Coste de construcción sobre rasante**", value=int(propuesta1_edifi[propuesta1_edifi["Nombre"]=="coste_sobrerasante"]["Valor"]))
            st.metric(label="**Coste de construcción bajo rasante**", value=int(propuesta1_edifi[propuesta1_edifi["Nombre"]=="coste_bajorasante"]["Valor"]))
            super_df = propuesta1_edifi[(propuesta1_edifi['Nombre'].str.startswith('super')) & (propuesta1_edifi['Nombre']!="super_viviendas")].drop("Nombre", axis=1)
            super_df["Valor"] = round(super_df["Valor"],1)
            st.plotly_chart(bar_plotly(super_df, "Distribución de superficie en m2", "Superficie en m2"), use_container_width=True, responsive=True)
        with center:
            propuesta2_edifi = propuesta_edifi_df[1]
            st.markdown(f'<div class="custom-box">PROPUESTA 2</div>', unsafe_allow_html=True)   
            st.metric(label="**Número de viviendas**", value=int(propuesta2_edifi[propuesta2_edifi["Nombre"]=="num_viviendas"]["Valor"]))
            st.metric(label="**Superficie de viviendas**", value=int(propuesta2_edifi[propuesta2_edifi["Nombre"]=="super_viviendas"]["Valor"]))
            st.metric(label="**Coste de construcción sobre rasante**", value=int(propuesta2_edifi[propuesta2_edifi["Nombre"]=="coste_sobrerasante"]["Valor"]))
            st.metric(label="**Coste de construcción bajo rasante**", value=int(propuesta2_edifi[propuesta2_edifi["Nombre"]=="coste_bajorasante"]["Valor"]))
            super_df = propuesta2_edifi[(propuesta2_edifi['Nombre'].str.startswith('super')) & (propuesta2_edifi['Nombre']!="super_viviendas")].drop("Nombre", axis=1)
            super_df["Valor"] = round(super_df["Valor"],1)
            st.plotly_chart(bar_plotly(super_df, "Distribución de superficie en m2", "Superficie en m2"), use_container_width=True, responsive=True)
        with right:
            propuesta3_edifi = propuesta_edifi_df[2]
            st.markdown(f'<div class="custom-box">PROPUESTA 3</div>', unsafe_allow_html=True)   
            st.metric(label="**Número de viviendas**", value=int(propuesta3_edifi[propuesta3_edifi["Nombre"]=="num_viviendas"]["Valor"]))
            st.metric(label="**Superficie de viviendas**", value=int(propuesta3_edifi[propuesta3_edifi["Nombre"]=="super_viviendas"]["Valor"]))
            st.metric(label="**Coste de construcción sobre rasante**", value=int(propuesta3_edifi[propuesta3_edifi["Nombre"]=="coste_sobrerasante"]["Valor"]))
            st.metric(label="**Coste de construcción bajo rasante**", value=int(propuesta3_edifi[propuesta3_edifi["Nombre"]=="coste_bajorasante"]["Valor"]))
            super_df = propuesta3_edifi[(propuesta3_edifi['Nombre'].str.startswith('super')) & (propuesta3_edifi['Nombre']!="super_viviendas")].drop("Nombre", axis=1)
            super_df["Valor"] = round(super_df["Valor"],1)
            st.plotly_chart(bar_plotly(super_df, "Distribución de superficie en m2", "Superficie en m2"), use_container_width=True, responsive=True)

if selected == "Análisis estático":
    left, right= st.columns((1,1))
    with left:
        selected_propuesta = st.radio("", ("Propuesta 1", "Propuesta 2", "Propuesta 3", "Comparativa"), horizontal=True)
    with right:
        max_trim = st.slider("**Número de trimestres de la operación**", 8, 16, 10)
    if selected_propuesta!="Comparativa":
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing_data = conn.read(worksheet="Propuesta_est" + selected_propuesta[-1], usecols=list(range(3)), ttl=5)
        existing_data = existing_data.dropna(how="all")

        for index, row in existing_data.iterrows():
            variable_name = row['Nombre']
            variable_value = (row['Valor'])
            exec(f"{variable_name} = {variable_value}")
        left, center, right= st.columns((1,1,1))
        with left:
            input_recursospropios = st.number_input("Recursos propios", min_value=0.0, max_value=999999999.0, value=input_recursospropios, step=1000.0)
            input_creditoconcedido = st.number_input("Crédito concedido",  min_value=0.0, max_value=999999999.0, value=input_creditoconcedido, step=1000.0)
            input_gastosconstitucion = st.number_input("Gastos de constitución",  min_value=0.0, max_value=999999999.0, value=input_gastosconstitucion, step=1000.0)
        with center:
            input_tipodeinteres = st.number_input("Tipo de interés",  min_value=0.0, max_value=999999999.0, value=input_tipodeinteres, step=1000.0)
            input_comisiondeapertura = st.number_input("Comisión de apertura",  min_value=0.0, max_value=999999999.0, value=input_comisiondeapertura, step=1000.0)
            input_comisiondecancelacion = st.number_input("Comisión de cancelación",  min_value=0.0, max_value=999999999.0, value=input_comisiondecancelacion, step=1000.0)
        with right:
            input_superficieconstruida = st.number_input("Superficie construida",  min_value=0.0, max_value=999999999.0, value=input_superficieconstruida, step=1000.0)
            input_costem2construido = st.number_input("Coste promedio del m2 construido",  min_value=0.0, max_value=999999999.0, value=input_costem2construido, step=1000.0)
            input_proporcioncompraplano = st.number_input("Proporción de compra sobre plano",  min_value=0.0, max_value=999999999.0, value=input_proporcioncompraplano, step=1000.0)
        
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
                conn.update(worksheet="Propuesta_est" + selected_propuesta[-1], data=result_df)
                st.success("¡Propuesta guardada!")
            conn = st.connection("gsheets", type=GSheetsConnection)
            existing_data = conn.read(worksheet="Propuesta_est" + selected_propuesta[-1], usecols=list(range(3)), ttl=5)
            existing_data = existing_data.dropna(how="all")
            with left:
                def treemap():
                    data = existing_data[["Parametro", "Valor"]][existing_data["Parametro"]!="INGRESOS POR VENTAS"].iloc[9:,:]
                    
                    # Calculate proportions
                    total_value = data['Valor'].sum()
                    data['Proportion'] = data['Valor'] / total_value * 100
                    
                    # Format proportion as percentage
                    data['Proportion'] = data['Proportion'].round(2).astype(str) + '%'

                    # Create the treemap
                    fig = px.treemap(
                        data,
                        names='Parametro',
                        values='Valor',
                        parents=[''] * len(data),
                        title="Distribució dels costos de la promoció",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        hover_data={'Proportion': True}  # Display proportion as hover data
                    )
                    fig.update_layout(height=600, width=1500, paper_bgcolor="#edf1fc", plot_bgcolor='#edf1fc')
                    return fig
                def sorted_barplot_with_proportions():
                    data = existing_data[["Parametro", "Valor"]][existing_data["Parametro"] != "INGRESOS POR VENTAS"].iloc[9:,:]
                    
                    # Calculate proportions
                    total_value = data['Valor'].sum()
                    data['Proportion'] = data['Valor'] / total_value * 100
                    
                    # Format proportion as percentage
                    data['Proportion'] = data['Proportion'].round(2).astype(str) + '%'

                    # Sort values by proportion
                    data = data.sort_values(by='Proportion', ascending=False)

                    # Create the bar plot
                    fig = px.bar(
                        data.sort_values(by='Proportion', ascending=False),
                        x='Parametro',
                        y='Valor',
                        title="Distribució dels costos de la promoció",
                        color='Parametro',
                        hover_data={'Proportion': True},  # Display proportion as hover data
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )

                    # Add annotations for labels
                    for i, row in data.iterrows():
                        fig.add_annotation(
                            x=row['Parametro'], y=row['Valor'],
                            text=row['Proportion'],
                            showarrow=False
                        )

                    fig.update_layout(height=600, width=1500, paper_bgcolor="#edf1fc", plot_bgcolor='#edf1fc')
                    return fig

                st.plotly_chart(sorted_barplot_with_proportions())
                st.plotly_chart(treemap())
        left, right = st.columns((1,1))
        with left:
            calcul_roe = round(((total_ingresos - total_gastos - total_fin)/input_recursospropios)*100, 1)
            calcul_roi = round(((total_ingresos - total_gastos)/total_gastos)*100, 1)
            renta_anual = round((((1+(calcul_roe/100))**(1/(max_trim/4)))-1)*100,1)
            def barplot_ratios():
                x_values = ['ROE', 'ROI', "Rentabilidad anualizada"]
                y_values = [calcul_roe, calcul_roi, renta_anual]
                labels = [f'{value:.1f}%' for value in y_values]  # Format values as percentage with two decimal places
                fig = go.Figure(data=[go.Bar(x=x_values, y=y_values, text=labels, textposition='auto', width=0.5)])  # Adjust width as needed
                fig.update_layout(title='RATIOS FINANCIERAS',
                                xaxis_title='Métrica',
                                yaxis_title='Porcentaje')
                return fig
            st.plotly_chart(barplot_ratios())
        with right:
            st.write("Result")
    if selected_propuesta=="Comparativa":
        num_propuesta_estatico = ["Propuesta_est1", "Propuesta_est2", "Propuesta_est3"]
        propuesta_estatico_df = []
        for i in num_propuesta_estatico:
            conn = st.connection("gsheets", type=GSheetsConnection)
            existing_data = conn.read(worksheet=i, usecols=list(range(3)), ttl=5)
            existing_data = existing_data.dropna(how="all")
            propuesta_estatico_df.append(existing_data)
        # def bar_plotly(df, title_main, title_y):
        #     fig = go.Figure()
        #     fig.add_trace(go.Bar(
        #         x=df["Elemento"],
        #         y=df["Valor"],
        #         name="m2",
        #         text=df["Valor"],  # Use the "Valor" column for labels
        #         textposition='auto'  # Position the labels automatically
        #     ))
        #     fig.update_layout(
        #         title=title_main,
        #         xaxis=dict(title=""),
        #         yaxis=dict(title=title_y, tickformat=",d"),
        #         paper_bgcolor="#edf1fc",
        #         plot_bgcolor='#edf1fc'
        #     )
        #     return fig
        def calculate_metrics(df):
            total_gastos = df[df["Nombre"].isin(["input_solar1", "input_solar2", "input_solar3", "input_edificacion2", "input_edificacion3", "input_edificacion4", "input_edificacion5", "input_edificacion6", "input_com1", "input_admin1"])]["Valor"].sum()
            total_fin = df[df["Nombre"].isin(["input_fin1", "input_fin2"])]["Valor"].sum()
            total_ingresos = df[df["Nombre"]=="input_ventas1"]["Valor"].sum()
            BAII_propuesta = total_ingresos - total_gastos #BAII
            BAI_propuesta = total_ingresos - total_gastos - total_fin #BAI
            return([total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta])
        def bar_plotly_finmetrics(total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta):
            values = [total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta]
            labels = ['Total Gastos', 'Total Gastos de Financiación', 'Total Ingresos', 'BAII', 'BAI']
            bar_labels = [f"{value}€" for value in values]
            fig = go.Figure([go.Bar(x=labels, y=values, text=bar_labels, textposition='auto')])
            fig.update_layout(title='Resultado financiero',
                            xaxis_title='',
                            yaxis_title='€')
            
            return fig

        left, center, right= st.columns((1,1,1))
        with left:
            propuesta1_est = propuesta_estatico_df[0]
            st.markdown(f'<div class="custom-box">PROPUESTA 1</div>', unsafe_allow_html=True)   
            # st.plotly_chart(bar_plotly(super_df, "Distribución de superficie en m2", "Superficie en m2"))
            total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta = calculate_metrics(propuesta1_est)
            st.metric(label="**Total ingresos**", value=int(total_ingresos))
            st.metric(label="**Total gastos**", value=int(total_gastos))
            st.metric(label="**Total gastos de financiación**", value=int(total_fin))
            st.metric(label="**BAII**", value=int(BAII_propuesta))
            st.metric(label="**BAI**", value=int(BAI_propuesta))
            st.plotly_chart(bar_plotly_finmetrics(total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta), use_container_width=True, responsive=True)

        with center:
            propuesta2_est = propuesta_estatico_df[1]
            st.markdown(f'<div class="custom-box">PROPUESTA 2</div>', unsafe_allow_html=True)   
            total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta = calculate_metrics(propuesta2_est)
            st.metric(label="**Total ingresos**", value=int(total_ingresos))
            st.metric(label="**Total gastos**", value=int(total_gastos))
            st.metric(label="**Total gastos de financiación**", value=int(total_fin))
            st.metric(label="**BAII**", value=int(BAII_propuesta))
            st.metric(label="**BAI**", value=int(BAI_propuesta))
            st.plotly_chart(bar_plotly_finmetrics(total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta), use_container_width=True, responsive=True)

        with right:
            propuesta3_est = propuesta_estatico_df[2]
            st.markdown(f'<div class="custom-box">PROPUESTA 3</div>', unsafe_allow_html=True)   
            total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta = calculate_metrics(propuesta3_est)
            st.metric(label="**Total ingresos**", value=int(total_ingresos))
            st.metric(label="**Total gastos**", value=int(total_gastos))
            st.metric(label="**Total gastos de financiación**", value=int(total_fin))
            st.metric(label="**BAII**", value=int(BAII_propuesta))
            st.metric(label="**BAI**", value=int(BAI_propuesta))
            st.plotly_chart(bar_plotly_finmetrics(total_gastos, total_fin, total_ingresos, BAII_propuesta, BAI_propuesta), use_container_width=True, responsive=True)


if selected == "Análisis dinámico":
    def date_to_quarter(date_str):
        date = datetime.strptime(str(date_str), '%Y-%m-%d')
        quarter = (date.month - 1) // 3 + 1
        year_quarter = f'{date.year}T{quarter}'
        return year_quarter
    def add_quarters(start_date, num_quarters):
        current_quarter = date_to_quarter(start_date)
        year, quarter = current_quarter.split('T')
        new_year = int(year) + (int(quarter) + num_quarters - 1) // 4
        new_quarter = (int(quarter) + num_quarters - 1) % 4 + 1
        return f"{new_year}T{new_quarter}"
    left_c, left, center, right, right_c = st.columns((1,1,1,1,1))
    with left:
        start_date =st.date_input("Fecha de inicio de la operación", value="today")
    with center:
        max_trim = st.slider("**Número de trimestres de la operación**", 8, 16, 10)
    with right:
        selected_propuesta = st.radio("", ("Propuesta 1", "Propuesta 2", "Propuesta 3"), horizontal=True)

    conn = st.connection("gsheets", type=GSheetsConnection)
    existing_data = conn.read(worksheet="Propuesta_din"+ selected_propuesta[-1], usecols=list(range(12)), ttl=5)
    existing_data = existing_data.dropna(how="all")
    start_quarter = date_to_quarter(start_date)
    st.write(f"Trimestre de inicio: {start_quarter}")
    quarters = []
    for i in range(max_trim):
        quarters.append(add_quarters(start_date, i))
    quarters.append("TOTAL")
    n_columns = st.columns(len(quarters)+1)
    n_elements = existing_data["Tesorería"].tolist()
    analisis_din = pd.DataFrame(index=n_elements)
    for i, input_col in enumerate(n_columns):
        column_data = []
        with input_col:
            if i == 0:
                for j, element in enumerate(n_elements):
                    value= st.text_input("", element, key=j*1000)
                    column_data.append(value)
            else:
                for j in range(len(n_elements)+1):
                    if j == 0:
                        value = st.write(quarters[i - 1])
                    else:
                        value = st.number_input("", value=existing_data.iloc[j-1,i], key=int(str(i) + str(j) + str(j+1)))
                        column_data.append(value)
            analisis_din[str(quarters[i-1])] = column_data
    analisis_din = analisis_din.reset_index().rename(columns= {"index":"Tesorería"})
    analisis_din["Total"] = analisis_din["TOTAL"] 
    analisis_din.drop("TOTAL", axis=1, inplace=True)
    submit_button = st.button(label="Guardar proposta")
    if submit_button:
        conn.update(worksheet="Propuesta_din"+ selected_propuesta[-1], data=analisis_din)
        st.success("¡Propuesta guardada!")
    st.table(analisis_din)
################################# ANALISI DE MERCADO DATOS PÚBLICOS (AHC) #########################################################

# @st.cache_resource
# def import_data():
#     maestro_mun = pd.read_excel("Maestro_MUN_COM_PROV.xlsx", sheet_name="Maestro")
#     DT_terr_def = pd.read_excel("DT_simple.xlsx", sheet_name="terr_q")
#     DT_terr_y_def = pd.read_excel("DT_simple.xlsx", sheet_name="terr_y")
#     DT_mun_def = pd.read_excel("DT_simple.xlsx", sheet_name="mun_q")
#     DT_mun_y_def = pd.read_excel("DT_simple.xlsx", sheet_name="mun_y")
#     return([DT_terr_def, DT_terr_y_def, DT_mun_def, DT_mun_y_def, maestro_mun])

# DT_terr, DT_terr_y, DT_mun, DT_mun_y, maestro_mun = import_data()
@st.cache_resource
def import_data(trim_limit, month_limit):
    with open('Censo2021.json', 'r') as outfile:
        list_censo = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    censo_2021= list_censo[0].copy()
    rentaneta_mun= list_censo[1].copy()
    rentaneta_mun = rentaneta_mun.applymap(lambda x: x.replace(".", "") if isinstance(x, str) else x)
    rentaneta_mun = rentaneta_mun.apply(pd.to_numeric, errors='ignore')
    censo_2021_dis= list_censo[2].copy()
    rentaneta_dis = list_censo[3].copy()
    rentaneta_dis = rentaneta_dis.applymap(lambda x: x.replace(".", "") if isinstance(x, str) else x)
    rentaneta_dis = rentaneta_dis.apply(pd.to_numeric, errors='ignore')
    with open('DT_simple.json', 'r') as outfile:
        list_of_df = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    DT_terr= list_of_df[0].copy()
    DT_mun= list_of_df[1].copy()
    DT_mun_aux= list_of_df[2].copy()
    DT_mun_aux2= list_of_df[3].copy()
    DT_mun_aux3= list_of_df[4].copy()
    DT_dis= list_of_df[5].copy()
    DT_terr_y= list_of_df[6].copy()
    DT_mun_y= list_of_df[7].copy()
    DT_mun_y_aux= list_of_df[8].copy()
    DT_mun_y_aux2= list_of_df[9].copy()
    DT_mun_y_aux3= list_of_df[10].copy()
    DT_dis_y= list_of_df[11].copy()
    DT_monthly= list_of_df[12].copy()
    DT_monthly["Fecha"] = DT_monthly["Fecha"].astype("datetime64[ns]")
    maestro_mun= list_of_df[13].copy()
    maestro_dis= list_of_df[14].copy()


    DT_monthly = DT_monthly[DT_monthly["Fecha"]<=month_limit]
    DT_terr = DT_terr[DT_terr["Fecha"]<=trim_limit]
    DT_mun = DT_mun[DT_mun["Fecha"]<=trim_limit]
    DT_mun_aux = DT_mun_aux[DT_mun_aux["Fecha"]<=trim_limit]
    DT_mun_aux2 = DT_mun_aux2[DT_mun_aux2["Fecha"]<=trim_limit]
    DT_mun_aux3 = DT_mun_aux3[DT_mun_aux3["Fecha"]<=trim_limit]
    DT_mun_pre = pd.merge(DT_mun, DT_mun_aux, how="left", on=["Trimestre","Fecha"])
    DT_mun_pre2 = pd.merge(DT_mun_pre, DT_mun_aux2, how="left", on=["Trimestre","Fecha"])
    DT_mun_def = pd.merge(DT_mun_pre2, DT_mun_aux3, how="left", on=["Trimestre","Fecha"])
    mun_list_aux = list(map(str, maestro_mun.loc[maestro_mun["ADD"] == "SI", "Municipi"].tolist()))
    mun_list = ["Trimestre", "Fecha"] + mun_list_aux
    muns_list = '|'.join(mun_list)
    DT_mun_def = DT_mun_def[[col for col in DT_mun_def.columns if any(mun in col for mun in mun_list)]]
    DT_dis = DT_dis[DT_dis["Fecha"]<=trim_limit]
    DT_mun_y_pre = pd.merge(DT_mun_y, DT_mun_y_aux, how="left", on="Fecha")
    DT_mun_y_pre2 = pd.merge(DT_mun_y_pre, DT_mun_y_aux2, how="left", on="Fecha")
    DT_mun_y_def = pd.merge(DT_mun_y_pre2, DT_mun_y_aux3, how="left", on="Fecha")    
    DT_mun_y_def = DT_mun_y_def[[col for col in DT_mun_y_def.columns if any(mun in col for mun in mun_list)]]

    return([DT_monthly, DT_terr, DT_terr_y, DT_mun_def, DT_mun_y_def, DT_dis, DT_dis_y, maestro_mun, maestro_dis, censo_2021, rentaneta_mun, censo_2021_dis, rentaneta_dis])

DT_monthly, DT_terr, DT_terr_y, DT_mun, DT_mun_y, DT_dis, DT_dis_y, maestro_mun, maestro_dis, censo_2021, rentaneta_mun, censo_2021_dis, rentaneta_dis = import_data("2024-01-01", "2023-12-01")

# @st.cache_resource
# def import_hogares():
#     hogares = pd.read_csv(path + "59543.csv", sep=";", decimal= ",", thousands=".").dropna(axis=0).drop("Total Nacional", axis=1)
#     hogares['Municipi'] = hogares['Municipios'].apply(lambda x: x.split(" ", 1)[-1])
#     hogares['Codi'] = hogares['Municipios'].apply(lambda x: x.split(" ", 1)[0])
#     hogares['PROV'] =  hogares["Codi"].str[0:2]
#     hogares = hogares[hogares["PROV"].isin(["08", '17', '43', '25'])].drop("PROV", axis=1)
#     hogares = hogares[hogares["Tamaño del hogar"]!="Total (tamaño del hogar)"].drop("Municipios", axis=1)
#     hogares["Tamaño"] = hogares["Tamaño del hogar"].str[0]
#     hogares = hogares.drop("Tamaño del hogar", axis=1)
#     return(hogares)
# hogares_mun = import_hogares()
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
    colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7"]
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
        paper_bgcolor = "#edf1fc",
        plot_bgcolor='#edf1fc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7"]
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
        paper_bgcolor = "#edf1fc",
        plot_bgcolor='#edf1fc'
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
        paper_bgcolor = "#edf1fc",
        plot_bgcolor='#edf1fc'
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
        paper_bgcolor = "#edf1fc",
        plot_bgcolor='#edf1fc'
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
    output_data = data_ori.set_index(["Any", "Trimestre"]).T#.dropna(axis=1, how="all")
    last_column_contains_all_nans = output_data.iloc[:, -1].isna().all()
    if last_column_contains_all_nans:
        output_data = output_data.iloc[:, :-1]
    else:
        output_data = output_data.copy()
    
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
    
#Defining years
max_year= 2024
available_years = list(range(2018,max_year))
index_year = 2023

############################################################# PESTAÑA ANALISIS DE MERCADO ########################################################

if selected=="Análisis de mercado":
    left, center,center_aux, right = st.columns((1,1,1,1))
    with left:
        selected_option = st.radio("**Àrea geográfica**", ("Provincias", "Municipios", "Distritos de Barcelona"), horizontal=True)
    with center:
        if selected_option=="Provincias":
            index_names = ["Preus", "Superfície", "Producció", "Compravendes"]
            selected_index = st.selectbox("**Selecciona un indicador:**", index_names)
        if selected_option!="Provincias":
            index_names_mun = ["Preus", "Superfície", "Producció", "Compravendes", "Definición de producto"]
            selected_index = st.selectbox("**Selecciona un indicador:**", index_names_mun)
    with center_aux:
        if selected_option=="Provincias":
            prov_names = ["Barcelona", "Girona", "Tarragona", "Lleida"]
            selected_geo = st.selectbox('**Selecciona una província:**', prov_names, index= prov_names.index("Barcelona"))      
        if selected_option=="Municipios":
            selected_mun = st.selectbox("**Selecciona un municipio:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"))
        if selected_option=="Distritos de Barcelona":
            selected_dis = st.selectbox("**Selecciona un distrito de Barcelona:**", maestro_dis["Districte"].unique())
    with right:
        max_year= 2024
        available_years = list(range(2018,max_year))
        index_year = 2023
        available_years = list(range(2018, datetime.now().year))
        if selected_index!="Definición de producto":
            selected_year_n = st.selectbox("**Selecciona un año:**", available_years, available_years.index(2023))
    if selected_option=="Provincias":
        if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")                
                with center:
                    try:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                with center:
                    try:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")

                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2008, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)     
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, max_year,["Any","Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True)     
            
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
            with center:
                st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)   
    if selected_option=="Municipios":
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_mun_pluri = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_mun_uni = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=0, delta="-100%")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=0, delta="-100%")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=0, delta="-100%")
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=0, delta="-100%")
            selected_columns_ini = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2008), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2008, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2008, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun_pluri, table_mun_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun_uni, table_mun_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=0, delta="-100%") 
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge nou**", value=0, delta="-100%") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
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
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value="n/a")
            with center:
                try:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="n/a")
            with right:
                try:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="n/a")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", 2005), use_container_width=True, responsive=True)
        if selected_index=="Definición de producto":
            def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
                table_n = table_n.reset_index()
                table_n["Any"] = table_n["Any"].astype(int)
                plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
                colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7"]
                traces = []
                for i, col in enumerate(plot_cat.columns):
                    trace = go.Bar(
                        x=plot_cat.index,
                        y=plot_cat[col]/1000,
                        name=col,
                        text=plot_cat[col], 
                        textfont=dict(color="white"),
                        marker=dict(color=colors[i % len(colors)]),
                    )
                    traces.append(trace)
                layout = go.Layout(
                    title=dict(text=title_main, font=dict(size=13)),
                    xaxis=dict(title="Any"),
                    yaxis=dict(title=title_y, tickformat=",d"),
                    legend=dict(x=0, y=1.15, orientation="h"),
                    paper_bgcolor = "#edf1fc",
                    plot_bgcolor='#edf1fc'
                )
                fig = go.Figure(data=traces, layout=layout)
                return fig
            def donut_plotly(table_n, selection_n, title_main, title_y):
                plot_cat = table_n[selection_n]
                plot_cat = plot_cat.set_index("Tamaño").sort_index()
                colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7", "#FFA07A"]
                traces = []
                for i, col in enumerate(plot_cat.columns):
                    trace = go.Pie(
                        labels=plot_cat.index,
                        values=plot_cat[col],
                        name=col,
                        hole=0.5,
                        marker=dict(colors=colors)
                    )
                    traces.append(trace)
                layout = go.Layout(
                    title=dict(text=title_main, font=dict(size=13)),
                    yaxis=dict(title=title_y),
                    legend=dict(x=0, y=1.15, orientation="h"),
                    paper_bgcolor="#edf1fc",
                    plot_bgcolor='#edf1fc'
                )
                fig = go.Figure(data=traces, layout=layout)
                return fig
            st.markdown(f'<div class="custom-box">DEMOGRAFÍA Y RENTA (2021)</div>', unsafe_allow_html=True)
            subset_tamaño_mun = censo_2021[censo_2021["Municipi"] == selected_mun][["1", "2", "3", "4", "5 o más"]]
            subset_tamaño_mun_aux = subset_tamaño_mun.T.reset_index()
            subset_tamaño_mun_aux.columns = ["Tamaño", "Hogares"]
            left, right = st.columns((1,1))
            with left:
                st.metric("Tamaño del hogar más frecuente", value=censo_2021[censo_2021["Municipi"]==selected_mun]["Tamaño_hogar_frecuente"].values[0])
                st.metric("Porcentaje de población extranjera", value=f"""{round(100 - censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_extranjera"].values[0],1)}%""")
                st.metric("Renta neta por hogar", value=rentaneta_mun["rentanetahogar_" + selected_mun].values[-1])
                st.plotly_chart(bar_plotly(rentaneta_mun.rename(columns={"Año":"Any"}).set_index("Any"), ["rentanetahogar_" + selected_mun], "Evolución anual de la renta media neta anual", "€", 2015), use_container_width=True, responsive=True)
            with right:
                st.metric("Tamaño medio del hogar", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Tamaño medio del hogar"].values[0],2)}""")
                st.metric("Porcentaje de población extranjera", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_extranjera"].values[0],1)}%""")
                st.metric("Porcentaje de población con educación superior", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Porc_Edu_superior"].values[0],1)}%""")
                st.plotly_chart(donut_plotly(subset_tamaño_mun_aux,["Tamaño", "Hogares"], "Distribución del número de miembros por hogar", "Hogares"), use_container_width=True, responsive=True)
            st.markdown(f'<div class="custom-box">CARACTERÍSTICAS DEL PARQUE TOTAL DE VIVIENDAS (2021)</div>', unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.metric("Porcentaje de viviendas en propiedad", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_propiedad"].values[0],1)}%""")
                st.metric("Porcentaje de viviendas principales", value=f"""{round(100 - censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_noprincipales_y"].values[0],1)}%""")
                st.metric("Edad media de las viviendas", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Edad media"].values[0],1)}""")
            with right:
                st.metric("Porcentaje de viviendas en alquiler", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_alquiler"].values[0], 1)}%""")
                st.metric("Porcentaje de viviendas no principales", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Perc_noprincipales_y"].values[0],1)}%""")
                st.metric("Superficie media de las viviendas", value=f"""{round(censo_2021[censo_2021["Municipi"]==selected_mun]["Superficie media"].values[0],1)}""")
            st.markdown(f'<div class="custom-box">TIPOLOGÍA I SUPERFICIES DE LOS VISADOS DE OBRA NUEVA Y CERTIFICADOS DE FIN DE OBRA</div>', unsafe_allow_html=True)
            table_ini_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_uni_", "iniviv_pluri_"], selected_mun), 2018, 2023,["Any","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars"]).sum(axis=0).reset_index().set_index("index")
            table_fin_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["finviv_uni_", "finviv_pluri_"], selected_mun), 2018, 2023,["Any","Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"]).sum(axis=0).reset_index().set_index("index")
            table_mun_pluri_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), 2018, 2023,["Any", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"]).sum(axis=0).reset_index().set_index("index")
            table_mun_uni_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), 2018, 2023, ["Any", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"]).sum(axis=0).reset_index().set_index("index")
            def donut_plotly(table_n, title_main, title_y):
                colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7", "#FFA07A"]
                traces = []
                for i, col in enumerate(table_n.columns):
                    trace = go.Pie(
                        labels=table_n.index,
                        values=table_n[col],
                        name=col,
                        hole=0.5,
                        marker=dict(colors=colors)
                    )
                    traces.append(trace)
                layout = go.Layout(
                    title=dict(text=title_main, font=dict(size=13)),
                    yaxis=dict(title=title_y),
                    legend=dict(x=0, y=1.15, orientation="h"),
                    paper_bgcolor="#edf1fc",
                    plot_bgcolor='#edf1fc'
                )
                fig = go.Figure(data=traces, layout=layout)
                return fig
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(donut_plotly(table_ini_mun_y, "Distribución por tipología de las viviendas visadas (2023-2018)",""))
                st.plotly_chart(donut_plotly(table_mun_pluri_y, "Distribución de superficies de las viviendas plurifamiliares visadas (2023-2018)",""))
            with right:
                st.plotly_chart(donut_plotly(table_fin_mun_y, "Distribución por tipología de las viviendas acabadas (2023-2018)",""))
                st.plotly_chart(donut_plotly(table_mun_uni_y, "Distribución de superficies de las viviendas unifamiliares visadas (2023-2018)",""))
    if selected_option=="Distritos de Barcelona":
        if selected_index=="Producció":
            min_year=2011
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            # table_dis_pluri = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            # table_dis_uni = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=0, delta="N/A")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=0, delta="N/A")
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=0, delta="N/A")           
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=0, delta="N/A")
            selected_columns_ini = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2014), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_dis[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_dis[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis_y.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2017), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), min_year, max_year,["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", 2017), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "var")}%""")
            with center:
                st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m2 útil", 2017), use_container_width=True, responsive=True)
        if selected_index=="Definición de producto":
            def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
                table_n = table_n.reset_index()
                table_n["Any"] = table_n["Any"].astype(int)
                plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
                colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7"]
                traces = []
                for i, col in enumerate(plot_cat.columns):
                    trace = go.Bar(
                        x=plot_cat.index,
                        y=plot_cat[col]/1000,
                        name=col,
                        text=plot_cat[col], 
                        textfont=dict(color="white"),
                        marker=dict(color=colors[i % len(colors)]),
                    )
                    traces.append(trace)
                layout = go.Layout(
                    title=dict(text=title_main, font=dict(size=13)),
                    xaxis=dict(title="Any"),
                    yaxis=dict(title=title_y, tickformat=",d"),
                    legend=dict(x=0, y=1.15, orientation="h"),
                    paper_bgcolor = "#edf1fc",
                    plot_bgcolor='#edf1fc'
                )
                fig = go.Figure(data=traces, layout=layout)
                return fig
            def donut_plotly(table_n, selection_n, title_main, title_y):
                plot_cat = table_n[selection_n]
                plot_cat = plot_cat.set_index("Tamaño").sort_index()
                colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7", "#FFA07A"]
                traces = []
                for i, col in enumerate(plot_cat.columns):
                    trace = go.Pie(
                        labels=plot_cat.index,
                        values=plot_cat[col],
                        name=col,
                        hole=0.5,
                        marker=dict(colors=colors)
                    )
                    traces.append(trace)
                layout = go.Layout(
                    title=dict(text=title_main, font=dict(size=13)),
                    yaxis=dict(title=title_y),
                    legend=dict(x=0, y=1.15, orientation="h"),
                    paper_bgcolor="#edf1fc",
                    plot_bgcolor='#edf1fc'
                )
                fig = go.Figure(data=traces, layout=layout)
                return fig
            st.markdown(f'<div class="custom-box">DEMOGRAFÍA Y RENTA (2021)</div>', unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                subset_tamaño_dis = censo_2021_dis[censo_2021_dis["Distrito"] == selected_dis][["1", "2", "3", "4", "5 o más"]]
                subset_tamaño_dis_aux = subset_tamaño_dis.T.reset_index()
                subset_tamaño_dis_aux.columns = ["Tamaño", "Hogares"]
                max_column = subset_tamaño_dis.idxmax(axis=1).values[0]
                st.metric("Tamaño del hogar más frecuente", value=max_column)
                st.metric("Porcentaje de población nacional", value=f"""{round(100 - censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_extranjera"].values[0]*100,1)}%""")
                st.metric("Renta neta por hogar", value=rentaneta_dis["rentahogar_" + selected_dis].values[-1])
                st.plotly_chart(bar_plotly(rentaneta_dis.rename(columns={"Año":"Any"}).set_index("Any"), ["rentahogar_" + selected_dis], "Evolución anual de la renta media neta anual", "€", 2015), use_container_width=True, responsive=True)
            with right:
                st.metric("Tamaño medio del hogar", value=f"""{censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Tamaño medio del hogar"].values[0]}""")
                st.metric("Porcentaje de población extranjera", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_extranjera"].values[0],1)}%""")
                st.metric("Porcentaje de población con educación superior", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_edusuperior"].values[0]*100,1)}%""")
                st.plotly_chart(donut_plotly(subset_tamaño_dis_aux,["Tamaño", "Hogares"], "Distribución del número de miembros por hogar", "Hogares"), use_container_width=True, responsive=True)
            st.markdown(f'<div class="custom-box">CARACTERÍSTICAS DEL PARQUE TOTAL DE VIVIENDAS (2021)</div>', unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.metric("Porcentaje de viviendas en propiedad", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_propiedad"].values[0],1)}%""")
                st.metric("Porcentaje de viviendas principales", value=f"""{round(100 - censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_noprincipales"].values[0],1)}%""")
                st.metric("Edad media de las viviendas", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Edad media"].values[0],1)}""")
            with right:
                st.metric("Porcentaje de viviendas en alquiler", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_alquiler"].values[0], 1)}%""")
                st.metric("Porcentaje de viviendas no principales", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Perc_noprincipales"].values[0],1)}%""")
                st.metric("Superficie media de las viviendas", value=f"""{round(censo_2021_dis[censo_2021_dis["Distrito"]==selected_dis]["Superficie Media"].values[0],1)}""")
            st.markdown(f'<div class="custom-box">TIPOLOGÍA I SUPERFICIES DE LOS VISADOS DE OBRA NUEVA Y CERTIFICADOS DE FIN DE OBRA</div>', unsafe_allow_html=True)
            table_ini_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_uni_", "iniviv_pluri_"], selected_dis), 2018, 2023,["Any","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars"]).sum(axis=0).reset_index().set_index("index")
            table_fin_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["finviv_uni_", "finviv_pluri_"], selected_dis), 2018, 2023,["Any","Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"]).sum(axis=0).reset_index().set_index("index")
            # table_dis_pluri_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_dis), 2018, 2023,["Any", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"]).sum(axis=0).reset_index().set_index("index")
            # table_dis_uni_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_dis), 2018, 2023, ["Any", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"]).sum(axis=0).reset_index().set_index("index")
            def donut_plotly(table_n, title_main, title_y):
                colors = ["#6495ED", "#7DF9FF",  "#87CEEB", "#A7C7E7", "#FFA07A"]
                traces = []
                for i, col in enumerate(table_n.columns):
                    trace = go.Pie(
                        labels=table_n.index,
                        values=table_n[col],
                        name=col,
                        hole=0.5,
                        marker=dict(colors=colors)
                    )
                    traces.append(trace)
                layout = go.Layout(
                    title=dict(text=title_main, font=dict(size=13)),
                    yaxis=dict(title=title_y),
                    legend=dict(x=0, y=1.15, orientation="h"),
                    paper_bgcolor="#edf1fc",
                    plot_bgcolor='#edf1fc'
                )
                fig = go.Figure(data=traces, layout=layout)
                return fig
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(donut_plotly(table_ini_dis_y, "Distribución por tipología de las viviendas visadas (2023-2018)",""))
                # st.plotly_chart(donut_plotly(table_mun_pluri_y, "Distribución de superficies de las viviendas plurifamiliares visadas (2023-2018)",""))
            with right:
                st.plotly_chart(donut_plotly(table_fin_dis_y, "Distribución por tipología de las viviendas acabadas (2023-2018)",""))
                # st.plotly_chart(donut_plotly(table_mun_uni_y, "Distribución de superficies de las viviendas unifamiliares visadas (2023-2018)",""))