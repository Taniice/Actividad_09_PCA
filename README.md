# Actividad_09
## A. PCA Dataset “Olivetti Faces Dataset” para crear un Sistema de reconocimiento de rostros basado en Eigenfaces

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

# C. Detección de anomalías en series temporales mediante Ensemble

## 1. Descripción

En este ejercicio se implementó un sistema de **detección de anomalías en una serie temporal** utilizando tres modelos de aprendizaje no supervisado:

* Isolation Forest
* KNN
* One-Class SVM

Posteriormente, se combinaron las predicciones mediante un **Ensemble basado en votación mayoritaria**, considerando una observación como anomalía cuando al menos **2 de los 3 modelos** la identificaron como tal.

El objetivo fue detectar comportamientos inusuales en una serie temporal de temperatura sin depender directamente de las etiquetas para realizar la detección.

---

## 2. Dataset

Se utilizó el dataset:

**Machine Temperature System Failure**, perteneciente al repositorio **Numenta Anomaly Benchmark (NAB)**.

El archivo utilizado fue:

`machine_temperature_system_failure.csv`

El dataset contiene principalmente:

* `timestamp`: fecha y hora de la medición.
* `value`: temperatura registrada.

La serie temporal comprende aproximadamente desde **diciembre de 2013 hasta febrero de 2014**.

---

## 3. Exploración y preparación de los datos

Primero se cargaron los datos y se convirtió la columna `timestamp` al formato de fecha y hora.

Posteriormente se realizó una visualización de la serie temporal para observar su comportamiento general.

Para los modelos se utilizó principalmente la variable:

`value`

La variable fue estandarizada mediante `StandardScaler`, permitiendo que los modelos trabajaran con los datos en una escala comparable.

---

## 4. Modelos utilizados

### 4.1 Isolation Forest

Isolation Forest identifica observaciones que pueden aislarse fácilmente del resto de los datos.

Configuración utilizada:

| Parámetro       | Valor |
| --------------- | ----: |
| `n_estimators`  |   300 |
| `contamination` | 0.005 |
| `random_state`  |    42 |

El modelo generó una predicción para cada observación y los valores identificados como `-1` fueron considerados anomalías.

**Anomalías detectadas: 111**

---

### 4.2 KNN

Para KNN se calculó la distancia de cada observación respecto a sus vecinos más cercanos.

Configuración utilizada:

| Parámetro     |          Valor |
| ------------- | -------------: |
| `n_neighbors` |             10 |
| Umbral        | Percentil 99.5 |

Las observaciones cuya distancia fue superior al percentil 99.5 fueron consideradas anomalías.

**Anomalías detectadas: 114**

---

### 4.3 One-Class SVM

One-Class SVM permite identificar observaciones que se encuentran fuera de la región considerada normal.

Configuración utilizada:

| Parámetro | Valor |
| --------- | ----: |
| `kernel`  |   RBF |
| `gamma`   |  0.01 |
| `nu`      | 0.001 |

El modelo identificó como anomalías las observaciones con predicción `-1`.

**Anomalías detectadas: 24**

---

## 5. Ensemble

Para combinar los tres modelos se utilizó una estrategia de **votación mayoritaria**.

Cada modelo genera una variable binaria:

* `1` → anomalía
* `0` → comportamiento normal

Posteriormente se sumaron los votos:

```text
Isolation Forest + KNN + One-Class SVM
```

Una observación fue considerada anomalía por el Ensemble cuando obtuvo **2 o más votos**.

```text
2 o 3 votos → ANOMALÍA
0 o 1 voto   → NORMAL
```

Esto permite reducir las detecciones que son producidas únicamente por un modelo.

---

## 6. Resultados

Los modelos obtuvieron los siguientes resultados:

| Modelo           | Anomalías detectadas |
| ---------------- | -------------------: |
| Isolation Forest |                  111 |
| KNN              |                  114 |
| One-Class SVM    |                   24 |
| **Ensemble**     |               **48** |

El Ensemble detectó **48 observaciones anómalas** utilizando la coincidencia de al menos dos modelos.

---

## 7. Análisis de las anomalías

Al visualizar la serie temporal con las detecciones del Ensemble, las anomalías se concentraron principalmente en eventos donde la temperatura presentó cambios poco frecuentes respecto al comportamiento general de la serie.

Se observaron principalmente:

<img width="1248" height="556" alt="image" src="https://github.com/user-attachments/assets/fdd723d4-1906-44de-a353-f04ef9e7292f" />


### Mediados de diciembre

Entre aproximadamente el **13 y 15 de diciembre**, se produjo una caída muy pronunciada de la temperatura, llegando cerca de 0. El Ensemble identificó múltiples puntos consecutivos durante este evento.

### Finales de diciembre

Entre aproximadamente el **24 y 25 de diciembre**, se identificó un pico aislado cercano a 108, diferente al comportamiento predominante de la serie.

### Febrero

Entre aproximadamente el **8 y 10 de febrero**, se presentó otra caída pronunciada de la temperatura, llegando aproximadamente a 25–30. Este comportamiento también fue detectado por el Ensemble.

Estas detecciones muestran que el modelo combinado identifica principalmente **cambios bruscos o comportamientos poco frecuentes** dentro de la serie temporal.

---

## 8. Conclusiones

Se implementó un sistema de detección de anomalías utilizando tres métodos no supervisados: **Isolation Forest, KNN y One-Class SVM**.

Los modelos produjeron diferentes cantidades de detecciones debido a que utilizan mecanismos distintos para identificar comportamientos inusuales.

Finalmente, se utilizó un **Ensemble de votación mayoritaria**, obteniendo 48 observaciones identificadas como anomalías.

La combinación de modelos permite utilizar el acuerdo entre diferentes métodos como criterio para realizar la detección, evitando depender de la predicción de un único algoritmo.

El resultado final muestra que las anomalías detectadas se concentran principalmente en eventos de cambios pronunciados de temperatura dentro de la serie temporal.

---

## 9. Tecnologías utilizadas

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Numenta Anomaly Benchmark (NAB)


