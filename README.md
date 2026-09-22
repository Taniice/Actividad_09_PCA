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
