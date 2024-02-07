![steam](https://github.com/maferhel/PI1H/blob/master/DATA%20AUX/steam.jpg)
<br />
# Proyecto MLOps: Sistema de Recomendación de Video Juegos

### Descripción del Proyecto
Este proyecto es la primer instancia fiinal de la fase de LABS del bootcamp de Henry, el que apunta a poner en práctica de habilidades técnicas adquiridas a lo largo de la cursada y las soft skills que cada uno posee, las que son necesarias reconocer para insertarse en el mercado laboral. La propuesta consiste en un caso de negocio real utilizando conjuntos de datos públicos de la industria de videojuegos,la plataforma en línea STEAM.

### Objetivo
El objetivo central de la propuesta es la creación de un modelo de Machine Learning (item-item). Para lograr esto del data set consumible se toma un item, en base a las similitudes de ese item con el resto que integran la base de datos, se recomiendan similares, utilizando para ello la herramienta de similitud del coseno. 

Asimismo, la elaboración de este producto involucra la realización de distintas tareas de Data Engineering como ser ETL, EDA, y construcción de una API hasta llegar a la implementación del modelo ML. En resumidas cuentas, se busca lograr un rápido desarrollo y tener un Producto Mínimo Viable (MVP) que mostrar al cliente que requiere el producto.<br />
<br />

## Etapas del Proyecto <br />
![DIAGRAMA](https://github.com/maferhel/PI1H/blob/master/DATA%20AUX/DIAGRAMA.jpg)  
<br />

**1. INGESTA DE DATOS** <br />
- En este primer paso disponibilice los datasets 'user_reviews.json.gz',  'steam_games.json.gz' y 'steam_games.json.gz' que se nos brindaron y pueden accederse a sus originales en **[DATA AUX](https://github.com/maferhel/PI1H/tree/master/DATA%20AUX).** y por medioo de los comandos display() e .info() accidí a su composición e información general para ver como cada data set estaba estructurado.<br />

**2. PREPROCESAMIENTO DE DATOS (EXPLORACIÓN PRELIMINAR)** <br />
- La finalidad de esta etapa es entender cual es la relación entre las tres tablas, ya que en el data frame "items" se encontrarían todas las claves o llaves a partir de las cuales se pueden establecer conexiones con las demás tablas. Por otro lado, la tabla "games" contiene todos los productos que integran el negocio y, por su parte, en la tabla "reviews" se han volcado datos a cerca de la experiencia que han tenido los usuarios en la utilización de esos productos.<br />
- A raiz de esta exploración se encontraron hallazgos preliminares que se relacionan con valores faltantes y composición en sí de los datas sets, que se fueron depurando a medida que se realizaban las consultas.<br />

**3. TRANSFORMACIÓN DE DATOS.** <br />
- En esta sección realicé transformaciones esenciales para cargar los conjuntos de datos con el formato adecuado. Estas transformaciones se llevaron a cabo con el propósito de optimizar tanto el rendimiento de la API como el entrenamiento del modelo. <br />
- Asimismo creé la columna **``` sentiment_analysis ```** aplicando análisis de sentimiento a las reseñas de los usuarios.Si bien, finalmente opté por aplicar definitivamente la libreria "textblob", previamente hice una comparación con los resultados arrojados por la biblioteca NLTK y mediante gráficos de torta representé los resultados las reseñas en negativas (valor '0'), neutrales (valor '1') o positivas (valor '2').<br />
- Finalmente opté por utilizar la librería "textblob" porque demostró una tendencia más "positiva" y, por ende útil para el futuro modelo de recomendación, ya que para ello necesesitamos que haya la mayor cantidad de recomendaciones positivas, o por lo menos neutrales, para que el modelo funcione mejor.<br />

**4. CARGA DE DATOS (DESARROLLO DE FUNCIONES).** <br />
- En vista de que los datos sean consumibles a través de una API se desarrollaron las funciones, que a continuación se enuncian, aclarando a su vez que los datasets fueron limitados a 5000 entradas, a los fines de obtener archivos livianos que se puedan deployar en los entornos correspondientes:<br />
  + Endpoint 1 (def developer): devuelve la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.<br />
  + Endpoint 2 (def userdata): la consulta retorna la cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.<br />
  + Endpoint 3 (def UserForGenre): su ejecución arroja el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.<br />
  + Endpoint 4 (def best_developer_year): su consulta devuelve el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado.<br />
  + Endpoint 5 (def developer_reviews_analysis): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.<br />

Para acceder a la funcionalidad completa de la API y explorar las recomendaciones de juegos, puedes visitar este enlace [URL de la API](https://pi1h.onrender.com/docs) En este sitio, encontrarás las diversas funciones desarrolladas.<br />
  
**5. PREPARACIÓN DEL MODELO DE RECOMENDACIÓN** <br />
- En esta etapa me avoque a trabajar sobre el data sets "df_UserForGenre", el que finalmente utilice para realizar el modelo de recomendación, concretamente el de recomendación de juego con una relación ítem-ítem, al que se le eliminaron las columnas innecesarias y se efectuaron las manipulaciones que eran apropiadas para disponibilizar una base de datos consumible por el modelo.<br />

**6. ANÁLISIS EXPLORATORIO DE DATOS (EDA)** <br />
- Depurada la base de datos, por medio de distintas erramientas de visualización, investigé relaciones entre variables, identifiqué outliers y busqué patrones interesantes en los datos.<br />

**7. ENTRENAMIENTO DEL MODELO DE RECOMENDACIÓN DE JUEGOS** <br />
- Aquí,a partir de la codificación de la columna "genres" realizada en la etapa 5, mediante la técnica de codificación one-hot para convertir los géneros en características binarias, teniendo presente la particularidad de esa columna, por ser una lista de cadenas de str, finalmente desarrollé la función para entrenar y consumir el modelo de ML, de recomendación de juegos.<br />

Como este readme es un esquema de lo que implicó el proyecyo, los invito a consultar el script [PI_NRO_1](https://github.com/maferhel/PI1H/blob/master/PI_NRO_1.ipynb) para profundizar en detalle cada tarea realizada. ¡Disfruta mientras navegas por el código!<br />


## Implementación de MLOps** <br />
Al igual que las funciones, el modelo de ML se puede consumir desde [URL de la API](https://pi1h.onrender.com/docs). <br />

## Video Explicativo** <br />
Grabé un video explicativo que muestra el funcionamiento de la API, consultas realizadas y una breve explicación de los modelos de ML utilizados [VIDEO](https://www.loom.com/share/edfee88d226e4b04a713f5a9cf8d8d4d).<br />
<br />


## Ejecutar la API (en su máquina local) <br />
1. Clonar el repositorio <br />
```
git@github.com:maferhel/PI1H.git
```
2. Crear entorno virtual<br />
```
python3 -m venv PI1H
```
3. Vaya al directorio del entorno virtual y actívelo<br />
- 3.1. Para Windows:
```
Scripts/activate
```
- 3.2. Para Linux/Mac:
```
bin/activate
```
4. Instalar los requerimientos<br />
```
pip install -r requirements.txt
```
5. Ejecute la API con uvicorn<br />
```
uvicorn main:app --reload
```


## Autor <br />
#### María Fernanda Helguero. <br />
Para cualquier duda/sugerencia/recomendación/mejora respecto al proyecto agradeceré que me la hagas saber, para ello contactame por [LinkedIn](https://www.linkedin.com/in/maria-fernanda-helguero-284087181/)<br />
