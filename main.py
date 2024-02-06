
from fastapi import FastAPI, Query,  HTTPException
import pandas as pd
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import requests
from io import BytesIO

app = FastAPI(
    # Esta línea se añadió para ignorar el favicon.ico porque daba error 404
    default_response_class_for_errors={404: HTTPException},
)

# CARGAR DATOS 

# Lista de URLs de los archivos en GitHub
urls = [
    'https://github.com/maferhel/PI1H/raw/master/DATA/df_UserForGenre.parquet',
    'https://github.com/maferhel/PI1H/raw/master/DATA/df_best_developer_year.parquet',
    'https://github.com/maferhel/PI1H/raw/master/DATA/df_developer_reviews_analysis.parquet',
    'https://github.com/maferhel/PI1H/raw/master/DATA/df_games_filt_def.parquet',
    'https://github.com/maferhel/PI1H/raw/master/DATA/df_muestramodelo.parquet',
    'https://github.com/maferhel/PI1H/raw/master/DATA/df_userdata.parquet'
]

dataframes = {}

# Iterar sobre las URLs y cargar los DataFrames
for url in urls:
    response = requests.get(url)
    response.raise_for_status()
    # Obtener el nombre del DataFrame a partir de la URL
    df_name = url.split('/')[-1].replace('.parquet', '')
    df = pd.read_parquet(BytesIO(response.content))
    # Asignar el DataFrame a una variable específica según su nombre
    dataframes[df_name] = df


# FUNCIONES 

#1.- def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 
def developer(desarrollador):
    df_games_filt_def = dataframes['df_games_filt_def']
    df_games_filt_def['release_date'] = pd.to_datetime(df_games_filt_def['release_date'])
    df_filtered = df_games_filt_def[df_games_filt_def['developer'] == desarrollador]
    df_filtered = df_filtered.dropna(subset=['release_date'])
    df_filtered['release_date'] = pd.to_datetime(df_filtered['release_date'], errors='coerce')

    result_data = {}

    for year in range(df_filtered['release_date'].min().year, df_filtered['release_date'].max().year + 1):
        year_items = len(df_filtered[df_filtered['release_date'].dt.year == year])
        year_free_items = len(df_filtered[(df_filtered['release_date'].dt.year == year) & (df_filtered['price'] == 0)])

        if year_items > 0:
            free_percentage = (year_free_items / year_items) * 100
        else:
            free_percentage = 0

        result_data[year] = {'Cantidad de Ítems': year_items, 'Contenido Free': f'{free_percentage:.0f}%'}

    return result_data

# Defino la ruta de la API y el método HTTP
@app.get("/developer/")
async def get_developer(desarrollador: str):
    resultado = developer(desarrollador)
    return resultado


# --------------------------------

#2.- def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
def userdata(User_id: str):
    df_userdata = dataframes['df_userdata']
    user_data = df_userdata[df_userdata['user_id'] == User_id]
    
    if user_data.empty:
        return {"message": f"User_id {User_id} inexistente", "money_spent": None, "recommendation_percentage": None, "item_count": None}
    
    money_spent = user_data['price'].sum() - user_data['discount_price'].sum()
    recommendation_percentage = (user_data['recommend'].mean()) * 100
    item_count = user_data['items_count'].values[0]
    
    return {"user_id": User_id, "money_spent": f"${money_spent:.2f}", "recommendation_percentage": f"{recommendation_percentage:.2f}%", "item_count": item_count}

# Defino la ruta de la API y el método HTTP
@app.get("/userdata/{User_id}")
async def get_userdata(User_id: str):
    result = userdata(User_id)
    return result


# --------------------------------

#3.- def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
def UserForGenre(genero: str, df_UserForGenre: pd.DataFrame):
    df_UserForGenre = dataframes['df_UserForGenre']
    df_UserForGenre = df_UserForGenre.dropna(subset=['genres'])
    df_expanded = df_UserForGenre.explode('genres')
    df_genre = df_expanded[df_expanded['genres'].str.contains(genero, case=False)]
    if df_genre.empty:
        return {"Usuario con más horas jugadas para " + genero: "No disponible", "Horas jugadas": []}
    user_hours = df_genre.groupby('user_id')['playtime_forever'].sum()
    user_most_played = user_hours.idxmax()
    df_genre['release_year'] = pd.to_datetime(df_genre['release_date']).dt.year
    hours_per_year = df_genre.groupby('release_year')['playtime_forever'].sum().reset_index()
    hours_per_year = hours_per_year.rename(columns={'release_year': 'Año', 'playtime_forever': 'Horas'})
    result = {
        f"Usuario con más horas jugadas para {genero}": user_most_played,
        "Horas jugadas": hours_per_year.to_dict(orient='records')
    }
    return result

# Defino la ruta de la API y el método HTTP
@app.get("/user_for_genre/")
async def get_user_for_genre(genero: str):
    result = UserForGenre(genero, dataframes['df_UserForGenre'])# Aquí debes proporcionar df_UserForGenre
    return result


# --------------------------------

    
#4.- def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
def best_developer_year(año: int):
    df_best_developer_year = dataframes['df_best_developer_year']
    juegos_del_año = df_best_developer_year[df_best_developer_year['release_date'].dt.year == año]
    juegos_recomendados = juegos_del_año[(juegos_del_año['recommend'] == True) & (juegos_del_año['sentiment_analysis'] == 2.0)]
    recomendaciones_por_desarrollador = juegos_recomendados['developer'].value_counts()
    desarrolladores_ordenados = recomendaciones_por_desarrollador.sort_values(ascending=False)
    top_3_desarrolladores = desarrolladores_ordenados.head(3)
    retorno = [{"Puesto 1": top_3_desarrolladores.index[0]}, 
               {"Puesto 2": top_3_desarrolladores.index[1]}, 
               {"Puesto 3": top_3_desarrolladores.index[2]}]
    
    return retorno

# Defino la ruta de la API y el método HTTP
@app.get("/best_developer_year/")
async def get_best_developer_year(año: int):
    resultado = best_developer_year(año)
    return resultado


# --------------------------------

#5.- def developer_reviews_analysis( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
def developer_reviews_analysis(desarrolladora: str, df_developer_reviews_analysis: pd.DataFrame) -> dict:
    df_developer_reviews_analysis = dataframes['df_developer_reviews_analysis']
    df_filtered = df_developer_reviews_analysis[df_developer_reviews_analysis['developer'] == desarrolladora]
    positive_count = (df_filtered['sentiment_analysis'] == 2.0).sum()
    negative_count = (df_filtered['sentiment_analysis'] == 0.0).sum()
    result = {desarrolladora: {'Positive': positive_count, 'Negative': negative_count}}
    
    return result

# Defino la ruta de la API y el método HTTP
@app.get("/developer_reviews_analysis/")
async def get_developer_reviews_analysis(desarrolladora: str):
    resultado = developer_reviews_analysis(desarrolladora, dataframes['df_developer_reviews_analysis']) # Asegúrate de proporcionar df_developer_reviews_analysis
    return resultado

# --------------------------------
# --------------------------------

# Modelo de Aprendizaje No Supervisado. RECOMENDACIÓN.

@app.get("/recomendacion/{producto_id}")
async def recomendacion_juego(producto_id: int, num_recomendaciones: int = 5) -> Tuple[str, List[str]]:
    df_muestramodelo = dataframes['df_muestramodelo']
    nombre_juego = df_muestramodelo[df_muestramodelo['item_id'] == producto_id]['item_name'].values[0]
    juego_caracteristicas = df_muestramodelo[df_muestramodelo['item_id'] == producto_id].iloc[:, 2:].values
    otros_juegos_caracteristicas = df_muestramodelo[df_muestramodelo['item_id'] != producto_id].iloc[:, 2:].values
    similarities = cosine_similarity(juego_caracteristicas, otros_juegos_caracteristicas)
    indices_similares = similarities.argsort()[0][-num_recomendaciones:][::-1]
    juegos_recomendados = df_muestramodelo.loc[df_muestramodelo.index[indices_similares], 'item_name'].tolist()
    
    return nombre_juego, juegos_recomendados


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)