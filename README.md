# Generación Automática de Configuraciones Visuales 

Autores: 

- Victor Manuel Cardentey Fundora C511
- Karla Olivera Hernández C511
- Amanda González Borrell C511

## Motivación

La generación automática de visualizaciones sobre un conjunto de datos se puede dividir en dos procesos: determinar una consulta de interés para el usuario y generar la configuración gráfica para visualizar los resultados de la consulta. En particular la selección de configuraciones gráficas es un problema que presenta dificultades para llegar a consenso entre expertos del dominio y los sistemas tradicionales que brindan solución a este problema utilizan enfoques basados en reglas. En años recientes se ha planteado la posibilidad de aplicar técnicas de *Machine Learning* ampliamente utilizadas en sistemas de recomendación tradicionales a la recomendación de configuraciones gráficas.

## Enfoque

La propuesta de este trabajo consiste en utilizar y comparar distintos modelos de *Machine Learning* en la tarea de selección de configuraciones gráficas. Esta tarea puede ser vista de forma simplificada como un problema de predicción.

>Dado un conjunto de vectores $v_1, v_2,...,v_k$ predecir el tipo de gráfico $g$ a utilizar y la permutación $v_{p1}, v_{p2}, ..., v_{pk}$ de los vectores iniciales que representa el orden que ocupan estos en los ejes del gráfico. 

Para la realización de esta propuesta se tienen como datos un conjunto de gráficos generados por usuarios de la plataforma [Plotly](https://plotly.com/) obtenidos a través de su [API](https://api.plot.ly/v2) los cuales contienen los vectores y configuraciones gráficas utilizadas para generarlos. 

## Plan de Trabajo

1. Recolección, limpieza y estructuración del dataset de gráficos utilizando la API de Plotly.
1. Selección de *features* de los vectores e implementación de un módulo de *feature extraction*.
1. Implementación de modelos de *Machine Learning*.
1. Experimentación y comparación de resultados.

## Trabajo relacionado

Este trabajo contribuye al Trabajo de Diploma en Ciencia de la Computación de Victor Manuel Cardentey Fundora en Visualización Inteligente Automática de Datos y el software obtenido puediese ser incorporado como visualizador en el sistema [LETO](https://leto-ai.github.io) para comprobar sus resultados dentro una aplicación de extracción de información.
