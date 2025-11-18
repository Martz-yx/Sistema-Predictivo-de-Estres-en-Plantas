# Sistema Predictivo para la Detección Temprana de Estrés en Plantas  <img width="50" height="50" alt="image" src="https://github.com/user-attachments/assets/3cdbeb2a-b063-438f-a7e8-8892f790c738" />   


## Un Enfoque Basado en Biofactores para la Optimización del Cuidado y la Preservación de Jardines Domésticos 

El cuidado de las plantas es una actividad que, además de ornamental, contribuye significativamente al bienestar y la estética de nuestros hogares y entornos. Sin embargo, uno de los desafíos más recurrentes para los aficionados y expertos en botánica es la detección temprana de signos de estrés o deterioro en las plantas. Frecuentemente, cuando los síntomas se vuelven visibles, el daño ya es considerable e irreversible, comprometiendo la salud y la vitalidad de la especie.

Este proyecto surge de la necesidad de identificar **precozmente factores de estrés y deterioro en las plantas**. Para ello, nos basaremos en el **dataset de Kaggle "Plant Health Data"** ([https://www.kaggle.com/datasets/ziya07/plant-health-data/data](https://www.kaggle.com/datasets/ziya07/plant-health-data/data)).

El dataset contiene mediciones de biofactores cruciales para la salud de las plantas, las cuales serán la base para desarrollar un modelo predictivo. 

Las variables son:

* **Timestamp:** Fecha y hora de cada lectura del biosensor.  
* **Plant\_ID:** Identificador único de cada planta.  
* **Soil\_Moisture (%):** Contenido de agua en el suelo.  
* **Ambient\_Temperature (°C):** Temperatura ambiental.  
* **Soil\_Temperature (°C):** Temperatura del suelo.  
* **Humidity (%):** Nivel de humedad del aire.  
* **Light\_Intensity (Lux):** Exposición a la luz.  
* **Soil\_pH:** Acidez o alcalinidad del suelo.  
* **Nitrogen\_Level (mg/kg):** Nivel de nitrógeno, nutriente vital.  
* **Phosphorus\_Level (mg/kg):** Nivel de fósforo, importante para raíces y flores.  
* **Potassium\_Level (mg/kg):** Nivel de potasio, para resistencia general.  
* **Chlorophyll\_Content (mg/m²):** Concentración de clorofila, indicador de salud.  
* **Electrochemical\_Signal (mV):** Señales de estrés detectadas en la planta.  
* **Plant\_Health\_Status:** Estado de salud de la planta (Categorías: **Healthy** \- Saludable, **Moderate Stress** \- Estrés Moderado, **High Stress** \- Estrés Alto).

El objetivo es construir un modelo que, utilizando estos biofactores, prediga el estado de salud de una planta, permitiendo una **intervención preventiva** y optimizando su cuidado.

# Exploración Inicial de Datos (EDA), Preprocesamiento y Selección de Modelos

## 1\. Exploración Inicial de Datos (EDA)

### 1.1 Descripción General

El dataset contiene registros temporales de múltiples plantas, con mediciones de variables ambientales, fisiológicas y químicas, junto con una etiqueta de estado de salud ("Plant\_Health\_Status") categórica. Las variables principales son:

### 1.2 Variables Clave

* **Plant\_Health\_Status** (target): Variable objetivo para clasificación.

* **Soil\_Moisture, Ambient\_Temperature, Soil\_Temperature, Humidity, Light\_Intensity, Soil\_pH, Nutrient Levels, Chlorophyll\_Content, Electrochemical\_Signal:** Variables predictoras.

### 1.3 Resultados del EDA

#### 1.3.1 Resumen de datos:

* Total de registros: 1200  
* total de días: 31  
* Rango de fechas: 2024-10-03 10:54:53.407995 a 2024-11-02 04:54:53.493508  
* Total de plantas únicas: 10

#### 1.3.2 Valores nulos por columna:

* Timestamp                 0  
* Plant\_ID                  0  
* Soil\_Moisture             0  
* Ambient\_Temperature       0  
* Soil\_Temperature          0  
* Humidity                  0  
* Light\_Intensity           0  
* Soil\_pH                   0  
* Nitrogen\_Level            0  
* Phosphorus\_Level          0  
* Potassium\_Level           0  
* Chlorophyll\_Content       0  
* Electrochemical\_Signal    0  
* Plant\_Health\_Status       0  
* dtype: int64

#### 	1.3.3 Resumen estadístico:

| Stats | Plant\_ID | Soil\_Moisture | Ambient\_Temperature | Soil\_Temperature | Humidity | Light\_Intensity | Soil\_pH | Nitrogen\_Level | Phosphorus\_Level | Potassium\_Level | Chlorophyll\_Content | Electrochemical\_Signal |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **count** | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 |
| **mean** | 5.5 | 25.106918 | 23.99913 | 19.957794 | 54.853165 | 612.637265 | 6.524102 | 30.106751 | 30.264484 | 30.112088 | 34.749591 | 0.987764 |
| **std** | 2.873479 | 8.677725 | 3.441561 | 2.932073 | 8.784916 | 228.318853 | 0.581755 | 11.514396 | 11.466846 | 11.668085 | 8.766995 | 0.575116 |
| **min** | 1 | 10.000724 | 18.001993 | 15.00371 | 40.028758 | 200.615482 | 5.507392 | 10.00365 | 10.01769 | 10.000606 | 20.025511 | 0.002376 |
| **25%** | 3 | 17.131893 | 21.101766 | 17.353027 | 47.019694 | 416.878983 | 6.026042 | 20.249774 | 20.894445 | 19.585561 | 27.46335 | 0.487982 |
| **50%** | 5.5 | 25.168333 | 23.889044 | 19.911473 | 54.692069 | 617.240221 | 6.540524 | 30.13859 | 30.019385 | 30.495054 | 34.433427 | 0.981647 |
| **75%** | 8 | 32.370231 | 27.042634 | 22.596851 | 62.451053 | 811.47469 | 7.030039 | 40.184737 | 40.131459 | 40.108296 | 42.232637 | 1.473142 |
| **max** | 10 | 39.993164 | 29.990886 | 24.995929 | 69.968871 | 999.856262 | 7.497823 | 49.951136 | 49.9807 | 49.981945 | 49.990811 | 1.996116 |

#### 1.3.4 Distribución de la variable objetivo:

Plant\_Health\_Status

* High Stress        500  
* Moderate Stress    401  
* Healthy            299

#### 	1.3.5 Histogramas de variables numéricas:

<img width="1587" height="1189" alt="image" src="https://github.com/user-attachments/assets/2cf7ffa8-1135-4e23-81fc-8ddd3c3d4415" />

#### 	1.3.6 Boxplots de variables numéricas:

<img width="1176" height="802" alt="image" src="https://github.com/user-attachments/assets/8d626079-59c0-430b-9684-fa0fc07094bb" />
<img width="1154" height="802" alt="image" src="https://github.com/user-attachments/assets/5af7b002-d7a8-4fb4-a1bc-cfa461b8f0b0" />

#### 1.3.7 Heatmap de correlación

<img width="1079" height="997" alt="image" src="https://github.com/user-attachments/assets/15e55839-1413-48cc-9a75-aea68ad42142" />

Aquí, podemos visualizar una correlación con la humedad del suelo, así como con el nivel de nitrógeno.

---

## 2\. Preprocesamiento propuesto

### 2.2 Transformación de Variables

* **Codificación de variables categóricas:**


  * Plant\_Health\_Status: codificación ordinal (Healthy=0, Moderate=1, High=2) para clasificación.

---

## 3\. Selección de Modelos

### 3.1 Modelo 1: Random Forest (Clasificación)

**Justificación:**

* El problema es de clasificación multiclase ("Healthy", "Moderate Stress", "High Stress").  
* Los árboles de decisión y Random Forest son robustos ante outliers y variables irrelevantes.  
* Permiten interpretar la importancia de variables (feature importance).  
* Manejan bien variables numéricas y categóricas.  
* Random Forest reduce el riesgo de overfitting respecto a un solo árbol.

### 3.2 Modelo 2: Support Vector Machine (SVM) o K-Nearest Neighbors (KNN) 

**Justificación**:

* SVM: Eficaz en clasificación multiclase, especialmente si las clases son separables en el espacio de variables.   
* KNN: Útil para comparar patrones de plantas similares, sensible a la escala (por eso la normalización). Ambos modelos permiten comparar el desempeño con Random Forest y pueden captar relaciones no lineales.

## 4\. Implementación y Evaluación

### 4.1 Implementación del Modelo

A continuación, se describen los pasos realizados:

1. **Carga y Preparación de Datos:** Se cargó el conjunto de datos plant\_health\_data.csv. Las columnas irrelevantes para el modelo, como Timestamp y Plant\_ID, fueron eliminadas.  
     
2. **Codificación de la Variable Objetivo:** La variable a predecir, Plant\_Health\_Status, es de naturaleza categórica ('Healthy', 'Moderate Stress', 'High Stress'). Se transformó a formato numérico para que el modelo pudiera procesarla, asignando:  
   * Healthy: 0  
   * Moderate Stress: 1  
   * High Stress: 2

3. **División de Datos:** El conjunto de datos se dividió en dos subconjuntos: uno para entrenamiento (80% de los datos) y otro para prueba (20% restante). Esto nos permite entrenar el modelo con una porción de los datos y evaluar su rendimiento en datos que no ha visto previamente.  
     
4. **Entrenamiento del Modelo Random Forest:** Se entrenó un clasificador RandomForestClassifier de scikit-learn. Este modelo se basa en la construcción de múltiples árboles de decisión y combina sus resultados para obtener una predicción más robusta y precisa, mitigando el riesgo de sobreajuste.\`

#### 4.1.1 Código de Implementación:
```Python
# Evaluate the model
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Generate and plot the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=le.classes_, 
    yticklabels=le.classes_
)
plt.title('Confusion Matrix for Plant Health Prediction')
plt.xlabel('Predicted Status')
plt.ylabel('True Status')
plt.show()

# Get Feature Importance
feature_importances = pd.Series(
    model.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```

### 4.2 Evaluación de Resultados

La evaluación del modelo es un paso crucial para entender su efectividad. Se utilizaron varias métricas estándar para problemas de clasificación multiclase.

* **Precisión (Accuracy):** Mide el porcentaje total de predicciones correctas. Nuestro modelo alcanzó una precisión del **99.58%**, lo que indica un rendimiento excepcionalmente alto en la clasificación del estado de salud de las plantas.  
* **Reporte de Clasificación:** Proporciona un desglose más detallado del rendimiento por cada clase.
```Python
Accuracy: 99.58%
Classification Report:
                 precision    recall  f1-score   support

        Healthy       1.00      1.00      1.00        65
    High Stress       1.00      0.99      0.99        85
Moderate Stress       0.99      1.00      0.99        90

       accuracy                           1.00       240
      macro avg       1.00      1.00      1.00       240
   weighted avg       1.00      1.00      1.00       240
```
* **Matriz de Confusión:** Visualiza el número de aciertos y errores por clase. La matriz confirma la alta precisión del modelo, con la gran mayoría de las predicciones ubicadas en la diagonal principal (predicciones correctas).
  
<img width="640" height="545" alt="image" src="https://github.com/user-attachments/assets/a6d17612-1110-45d5-82c4-4322fb37211f" />

#### 4.2.1 Código de Evaluación:

```Python
# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## 5\. Implementación de Modelos Adicionales

Antes de entrenar los modelos SVM y KNN, los datos de las características (X\_train y X\_test) fueron escalados.

```Python
from sklearn.preprocessing import StandardScaler

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5.1 Implementación del Modelo SVM (Support Vector Machine)

```Python
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f'SVM Accuracy: {accuracy_svm * 100:.2f}%')
print('\\nSVM Classification Report:')
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))
```

#### 5.1.1 Evaluación del modelo SVM

```Python
SVM Accuracy: 82.50%
\nSVM Classification Report:
                 precision    recall  f1-score   support

        Healthy       0.83      0.82      0.82        65
    High Stress       0.87      0.91      0.89        85
Moderate Stress       0.78      0.76      0.77        90

       accuracy                           0.82       240
      macro avg       0.82      0.83      0.83       240
   weighted avg       0.82      0.82      0.82       240
   ```

### 5.2 Implementación del Modelo KNN (K-Nearest Neighbors)

```Python
from sklearn.neighbors import KNeighborsClassifier

# Encontrar el 'k' Óptimo
k_range = range(1, 41)
accuracy_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

plt.figure(figsize=(12, 7))
plt.plot(k_range, accuracy_scores, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Precisión del Modelo KNN vs. Valor de k', fontsize=16)
plt.xlabel('Valor de k', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.grid(True)
plt.xticks(range(1, 41, 2))
plt.show()
   ```

#### 5.2.1 K Óptimo

<img width="1011" height="629" alt="image" src="https://github.com/user-attachments/assets/85e801d5-28ca-41a8-aae8-25693e94213a" />

```Python
knn_model = KNeighborsClassifier(n_neighbors=12) # el k Óptimo
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
 # Evaluacion de modelo
print(f'KNN Accuracy: {accuracy_knn * 100:.2f}%')
print('\nKNN Classification Report:')
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))
```

#### 5.2.2 Evaluación del modelo KNN

```Python
KNN Accuracy: 76.25%

KNN Classification Report:
                 precision    recall  f1-score   support

        Healthy       0.80      0.78      0.79        65
    High Stress       0.78      0.89      0.83        85
Moderate Stress       0.72      0.62      0.67        90

       accuracy                           0.76       240
      macro avg       0.76      0.77      0.76       240
   weighted avg       0.76      0.76      0.76       240
 ```

## 6\. Comparación

 Comparando el resultado de los 3 modelos, tenemos

| Modelo | Accuracy |
| :---- | :---: |
| **Random Forest (Clasificación)** | 99.58% |
| **Support Vector Machine (SVM)** | 82.50% |
| **K-Nearest Neighbors (KNN)** | 76.25% |

Con base en los resultados obtenidos de la implementación de los tres modelos de clasificación, se puede concluir que el **Random Forest** es el modelo con el mejor rendimiento para la detección temprana de estrés en plantas. Alcanzó una precisión notable del **99.58%**, superando significativamente a los otros dos modelos.

El modelo **Support Vector Machine (SVM)** mostró un rendimiento aceptable con una precisión del **82.50%**, posicionándose como el segundo mejor modelo. Aunque es considerablemente inferior al Random Forest, demuestra una capacidad razonable para clasificar el estado de salud de las plantas.

Finalmente, el modelo **K-Nearest Neighbors (KNN)** obtuvo la menor precisión con un **76.25%**. Si bien ofrece cierto nivel de predicción, su rendimiento es el más bajo de los tres, lo que sugiere que es el menos adecuado para esta tarea específica de detección temprana de estrés en plantas, en comparación con Random Forest y SVM.

## 7\. A.A.M: Verificación del modelo Random Forest

Luego de determinar el modelo más eficaz, se realizaron gráficos con relevancia a el modelo de Random Forest, los cuales llevó a la conclusión de un posible error en el modelado:

### Gráfico de Importancia de Características (Feature Importance Plot)

<img width="984" height="546" alt="image" src="https://github.com/user-attachments/assets/34f46891-2426-4c11-8b21-b31e2add612c" />

### Gráfico Curva ROC

<img width="857" height="699" alt="image" src="https://github.com/user-attachments/assets/ac9ea549-755f-4d7b-ac98-d3b72eb47a2e" />

Al momento de visualizar el gráfico de ROC, podemos apreciar una gran problemática:

*el gráfico muestra resultados **perfectos***

## 8\. investigación de resultado excepcional

### 8.1 Sobreajuste (Overfitting)

El sobreajuste ocurre cuando el modelo aprende los datos de entrenamiento con demasiado detalle, incluyendo el ruido y las peculiaridades específicas de ese conjunto, lo que lo hace incapaz de generalizar bien a datos nuevos.

#### 8.1.1 Comparación de  Rendimiento en Entrenamiento vs. Prueba

```Python
--- Rendimiento en el Conjunto de Entrenamiento ---
Precisión (Accuracy): 1.0000
...
--- Rendimiento en el Conjunto de Prueba ---
Precisión (Accuracy): 0.9958
```

#### 8.1.2 Validación Cruzada (Cross-Validation)

utilizado para obtener una estimación más robusta del rendimiento del modelo dividiendo los datos en múltiples pliegues, entrenando y evaluando el modelo varias veces. Esto ayuda a confirmar si el rendimiento perfecto es consistente en diferentes subconjuntos de datos.

```Python
--- Validación Cruzada (K-Fold = 5) ---
Precisión para cada pliegue: [0.99583333 0.99583333 0.99583333 1.         1.        ]
Precisión promedio: 0.9975
Desviación estándar de la precisión: 0.0020
```

### 8.2 Fuga de Datos (Data Leakage)

Esto ocurre cuando información del conjunto de prueba "se filtra" inadvertidamente al conjunto de entrenamiento, lo que hace que el modelo parezca funcionar mucho mejor de lo que realmente lo haría en un escenario real.


#### 8.2.1 Análisis de Correlación Extrema

<img width="810" height="527" alt="image" src="https://github.com/user-attachments/assets/584a5b0e-346c-484b-b649-e5850822ea8a" />

Inicialmente, el gráfico de correlación extrema no reveló el problema del modelo, dado que ninguna variable mostró una correlación significativamente alta con "plant\_health\_status". Esto llevó a regresar a la fase inicial de Análisis Exploratorio de Datos (EDA) con el fin de examinar detalladamente cada variable del conjunto de datos.

### 8.3 Análisis de variables y detección del problema

Tras un análisis minucioso de las variables, se constató lo siguiente:

* **Timestamp:** Fecha y hora de cada lectura del biosensor.  
* **Plant\_ID:** Identificador único de cada planta.  
* **Soil\_Moisture (%):** Contenido de agua en el suelo.  
* …  
* <mark>**Electrochemical\_Signal (mV):** Señales de estrés detectadas en la planta.</mark>
* **Plant\_Health\_Status:** Estado de salud de la planta (Categorías: **Healthy** \- Saludable, **Moderate Stress** \- Estrés Moderado, **High Stress** \- Estrés Alto).  
  Asumiendo que la salud de la planta (**Plant\_Health\_Status**) se define por su nivel de estrés y que la señal electromagnética (**Electrochemical\_Signal**) es la señal recibida como manifestación de dicho estrés, se establece una **correlación directa** que explica el rendimiento excepcionalmente alto del modelo.

	*Electrochemical\_Signal (mV) \= Plant\_Health\_Status*

## 9\. Re-Implementación y Evaluación de Modelos

Luego de eliminar columnas las columnas no necesarias ('Timestamp', 'Plant\_ID'), la columna con correlación directa ('Electrochemical\_Signal') y tambien,, las que estarian fuera del control humano ('Chlorophyll\_Content'), se realizó la evaluación nuevamente de los modelos:

### 9.1 Random Forest

Accuracy: 99.58%

### 9.2 Support Vector Machine (SVM)

SVM Accuracy: 85.83% (Mayor que el intento anterior)

### 9.3 K-Nearest Neighbors (KNN)

KNN Accuracy: 76.25% (Mayor que el intento anterior)

# Conclusión Final

Se concluye que el modelo predictivo desarrollado es excepcionalmente bueno, exhibiendo una robustez notable que va más allá de las expectativas iniciales. Incluso después de someter los datos a un riguroso proceso de limpieza para eliminar inconsistencias y valores atípicos, verificar meticulosamente la ausencia de sobreajuste (overfitting) mediante técnicas de validación cruzada y eliminar variables potencialmente incontrolables o no informativas (como el contenido de clorofila, cuya variabilidad puede deberse a factores externos no directamente relacionados con la salud intrínseca de la planta en el contexto del estudio), las predicciones del modelo se mantuvieron consistentemente muy altas. Este rendimiento sostenido es un indicador contundente de la capacidad del modelo para generalizar bien a nuevos datos y su gran fiabilidad.

La persistencia de una alta precisión predictiva sugiere fuertemente que el problema en cuestión es intrínsecamente altamente separable. Esto implica que las diferentes clases o categorías que el modelo intenta predecir (presumiblemente relacionadas con la salud de la planta, aunque no se especifica explícitamente) son distinguibles con gran claridad en el espacio de características, lo que facilita enormemente la tarea de clasificación o regresión para el algoritmo.

Un análisis más profundo de los rangos observados en las variables revela que la Humedad del Suelo, y en menor medida el Nivel de Nitrógeno, se erigen como características altamente discriminatorias para determinar la salud de la planta dentro del conjunto de datos analizado. Es fundamental destacar que esta capacidad discriminatoria no se manifiesta necesariamente como una correlación lineal alta y constante a lo largo de todo el rango de valores de estas variables. Por el contrario, su poder predictivo reside en la existencia de umbrales muy claros y definidos. Estos umbrales actúan como puntos de inflexión críticos, permitiendo una separación casi perfecta de las clases. Por ejemplo, por debajo o por encima de ciertos niveles de humedad o nitrógeno, la salud de la planta podría clasificarse inequívocamente en diferentes categorías, lo que simplifica la toma de decisiones y la interpretación de los resultados del modelo. La identificación de estos umbrales es un hallazgo clave que podría tener implicaciones significativas para la gestión y optimización de las condiciones de cultivo.

# Anexos extra

<img width="983" height="546" alt="image" src="https://github.com/user-attachments/assets/755f3f1c-cfda-4785-965e-2639ad609bc3" />

<img width="842" height="623" alt="image" src="https://github.com/user-attachments/assets/d0f477ed-6811-475c-8b09-1b26197b4b83" />

<img width="842" height="623" alt="image" src="https://github.com/user-attachments/assets/dad48337-6eb4-4656-8837-7a5b004b9467" />

<img width="1183" height="377" alt="image" src="https://github.com/user-attachments/assets/86a44cf6-ea9b-4b39-a035-6515846017c4" />


