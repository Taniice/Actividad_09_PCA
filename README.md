# Actividad_09_PCA
## A.  Dataset “Olivetti Faces Dataset” para crear un Sistema de reconocimiento de rostros basado en Eigenfaces

Se desarrolló un sistema de reconocimiento facial utilizando el dataset **Olivetti Faces** de Scikit-learn, mediante la técnica Eigenfaces basada en PCA y el clasificador SVM con kernel RBF, sin utilizar redes neuronales.

Se aplicó PCA para reducir la dimensionalidad de las imágenes faciales de 4096 píxeles a 150 componentes principales, conservando características relevantes para el reconocimiento de rostros. Posteriormente, SVM fue entrenado para clasificar las identidades de las 40 personas del dataset.
<img width="1167" height="495" alt="image" src="https://github.com/user-attachments/assets/9f8ee1eb-1549-4c7b-8a04-736b8b3539c8" />


El modelo fue evaluado con 100 imágenes de prueba, obteniendo los siguientes resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | 98% |
| Macro Average - Precision | 0.99 |
| Macro Average - Recall | 0.98 |
| Macro Average - F1-score | 0.98 |
| Weighted Average - F1-score | 0.98 |

<img width="404" height="448" alt="image" src="https://github.com/user-attachments/assets/0feb81dc-7110-4ce4-9ff2-eb9ea2801ce6" />


Los resultados muestran que el sistema logró identificar correctamente el 98% de las imágenes de prueba, evidenciando un alto nivel de reconocimiento en este conjunto de datos.

En conclusión, la combinación de PCA (Eigenfaces) y SVM permitió implementar un sistema de reconocimiento facial eficaz mediante técnicas tradicionales de Machine Learning, cumpliendo con el requisito de no utilizar redes neuronales.


## B. Sistema de recomendación de artículos mediante Embeddings

## Descripción

Se desarrolló un sistema de recomendación de noticias cuyo objetivo es encontrar artículos que tengan un contenido **semánticamente similar** al artículo que está leyendo un usuario.

Para este ejercicio se utilizó el dataset **AG News**, que contiene noticias clasificadas en cuatro categorías: `World`, `Sports`, `Business` y `Sci/Tech`.

En lugar de utilizar una representación tradicional basada únicamente en palabras, se utilizaron **embeddings**, que permiten representar el significado de los textos mediante vectores numéricos.

---

### 1. Carga del dataset

Se cargó el dataset AG News utilizando `Hugging Face` y se combinaron los conjuntos de entrenamiento y prueba, obteniendo un total de **127,600 artículos**.

Posteriormente, se seleccionaron **10,000 artículos** para realizar el experimento, debido a que generar embeddings para todo el dataset requiere mayor cantidad de memoria y tiempo de procesamiento en Google Colab.

### 2. Conversión de categorías

Las etiquetas numéricas originales del dataset fueron convertidas a nombres de categorías:

| Etiqueta | Categoría |
| -------: | --------- |
|        0 | World     |
|        1 | Sports    |
|        2 | Business  |
|        3 | Sci/Tech  |

### 3. Generación de embeddings

Para representar los artículos se utilizó el modelo:

`all-MiniLM-L6-v2`

Este modelo pertenece a `Sentence Transformers` y transforma cada texto en un vector numérico de **384 dimensiones**.

De esta manera, artículos que poseen contenidos relacionados pueden quedar representados mediante vectores cercanos, aunque no utilicen exactamente las mismas palabras.

### 4. Cálculo de similitud

Para determinar qué artículos son similares se utilizó la **similitud del coseno**.

La comparación se realiza entre el vector del artículo consultado y los vectores de los 10,000 artículos disponibles.

Un valor mayor de similitud indica que los embeddings se encuentran más próximos en el espacio semántico.

### 5. Generación de recomendaciones

Para cada artículo consultado:

1. Se obtiene su embedding.
2. Se compara con los embeddings de los demás artículos.
3. Se calcula la similitud del coseno.
4. Se ordenan los artículos desde el más similar hasta el menos similar.
5. Se seleccionan los **5 artículos con mayor similitud**.

Además, se excluye el propio artículo consultado para evitar que aparezca como recomendación.

---

## Prueba con un artículo nuevo

Se utilizó como entrada el siguiente texto:

> Technology companies are developing new artificial intelligence systems that can improve productivity, automate tasks and transform the way businesses use software and digital services.

El texto está relacionado con inteligencia artificial, productividad, automatización, software y servicios digitales.

El sistema buscó dentro de los 10,000 artículos aquellos cuyos embeddings presentaran mayor similitud con este contenido.

### Resultados

| Recomendación | Categoría | Similitud coseno |
| ------------: | --------- | ---------------: |
|             1 | Sci/Tech  |           0.4935 |
|             2 | Sci/Tech  |           0.4889 |
|             3 | Sci/Tech  |           0.4502 |
|             4 | Sci/Tech  |           0.4433 |
|             5 | Sci/Tech  |           0.4394 |

---

## Análisis de resultados

Los cinco artículos recomendados pertenecen a la categoría **Sci/Tech**, lo cual es coherente con el contenido del artículo utilizado como consulta.

La primera recomendación obtuvo una similitud de **0.4935**, siendo el artículo más cercano entre los 10,000 artículos utilizados para la búsqueda.

Los artículos recomendados tratan temas relacionados con tecnología, empresas de tecnología, sistemas informáticos, estrategias digitales y servicios tecnológicos. Esto muestra que el modelo no necesita encontrar exactamente las mismas palabras para identificar una relación temática.

Por ejemplo, el artículo de entrada habla de **inteligencia artificial, productividad, automatización y software**, mientras que algunas recomendaciones hablan de **tecnología informática, empresas de tecnología y transformación de estrategias empresariales**. Aunque las palabras no son idénticas, existe una relación semántica entre los contenidos.

Señalar que el valor de similitud coseno es un valor utilizado para medir qué tan cercanos son sus embeddings en el espacio vectorial.
