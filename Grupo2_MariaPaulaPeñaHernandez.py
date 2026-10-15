#Se sube a GitHub ya que el tamaño del archivo no permitió su publicacion en Ude
# ==============================================================================
# Grupo 2
# Estudiantes:
# - María Paula Peña Hernández
# - Samuel David Velez Perez
# Unidad 2 - Taller Evaluativo: Introducción a la computación en Bioingeniería
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. LECTURA Y EXPLORACIÓN DE LOS DATOS CLÍNICOS
# ------------------------------------------------------------------------------

# Cargar el archivo clinical_data.csv
archivo_csv = 'clinical_data.csv'
df_clinico = pd.read_csv(archivo_csv)

print("=== 1.1 ESTRUCTURA DEL DATAFRAME ===")
print(f"Dimensiones (filas, columnas): {df_clinico.shape}")
print("\nTipos de datos por columna:")
print(df_clinico.dtypes)

print("\nValores faltantes por columna:")
print(df_clinico.isnull().sum())

print("\n=== 1.2 VISTA PREVIA DE LOS DATOS ===")
print(df_clinico.head())

# Identificar columnas numéricas
cols_numericas = ['Edad', 'Frecuencia_Cardiaca', 'PAM', 'Glucosa']

print("\n=== 1.3 ESTADÍSTICAS DESCRIPTIVAS BÁSICAS ===")
# Cálculo explícito de Media, Desviación Estándar, Mínimo y Máximo
estadisticas = df_clinico[cols_numericas].agg(['mean', 'std', 'min', 'max']).T
estadisticas.columns = ['Media', 'Desviación Estándar', 'Mínimo', 'Máximo']
print(estadisticas)

# ------------------------------------------------------------------------------
# 1.4 GRAFICACIÓN EXPLORATORIA DE LOS DATOS CLÍNICOS
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de Edad
axes[0].hist(df_clinico['Edad'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribución de Edad de los Pacientes', fontsize=12)
axes[0].set_xlabel('Edad (Años)')
axes[0].set_ylabel('Frecuencia')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Boxplot de Frecuencia Cardíaca segmentado por Diagnóstico
datos_sano = df_clinico[df_clinico['Diagnóstico'] == 'Sano']['Frecuencia_Cardiaca']
datos_taqui = df_clinico[df_clinico['Diagnóstico'] == 'Taquicardia']['Frecuencia_Cardiaca']


bp = axes[1].boxplot([datos_sano, datos_taqui], tick_labels=['Sano', 'Taquicardia'], patch_artist=True)

# Personalizar colores de las cajas del boxplot
colores = ['#66c2a5', '#fc8d62']
for patch, color in zip(bp['boxes'], colores):
    patch.set_facecolor(color)

axes[1].set_title('Frecuencia Cardíaca por Diagnóstico', fontsize=12)
axes[1].set_xlabel('Diagnóstico')
axes[1].set_ylabel('Frecuencia Cardíaca (LPM)')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ==============================================================================
# 2. PROCESAMIENTO DE SEÑALES ECG SIMULADAS
# ==============================================================================

import numpy as np
import scipy.io
from scipy.signal import iirnotch, filtfilt

# ------------------------------------------------------------------------------
# 2.1 DEFINICIÓN DE LA CLASE Signal
# ------------------------------------------------------------------------------

class Signal:
    def __init__(self, data, label, fs=500):
        # Atributo privado
        self.__data = np.array(data)
        # Atributos públicos
        self.label = label
        self.fs = fs

    # Getter para acceder al atributo privado 
    def get_data(self):
        return self.__data

    def filtrar_notch(self, freq_corte=60.0, q=30.0):
    # Aplica un filtro Notch (rechaza banda) a 60 Hz
        w0 = freq_corte / (self.fs / 2.0)
        b, a = iirnotch(w0, q)
        if self.__data.ndim == 2:
            return filtfilt(b, a, self.__data, axis=-1)
        return filtfilt(b, a, self.__data)

    def recortar(self, canal=0, inicio=0, fin=1000):
    # Selecciona un canal y un rango de muestras
        if self.__data.ndim == 2:
            return self.__data[canal, inicio:fin]
        return self.__data[inicio:fin]

    def calcular_estadisticas(self, segmento):
    #Calcula la media y la desviación estándar del segmento recortado
        media = np.mean(segmento)
        desviacion = np.std(segmento)
        return media, desviacion

    def graficar_segmento(self, segmento_original, segmento_filtrado, canal=0, inicio=0, fin=1000):
       # Grafica el segmento original vs. filtrado.
        tiempo = np.arange(inicio, fin) / self.fs

        plt.figure(figsize=(10, 4))
        plt.plot(tiempo, segmento_original, label='Original (Con ruido)', color='gray', alpha=0.6)
        plt.plot(tiempo, segmento_filtrado, label='Filtrada (Notch 60Hz)', color='blue' if self.label == 'Sano' else 'red')
        plt.title(f'Señal ECG - Paciente {self.label} (Canal {canal})')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Amplitud (mV)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------------------
# 2.2 CARGA DE DATOS E INSTANCIACIÓN DE OBJETOS
# ------------------------------------------------------------------------------

# Carga archivos .mat usando scipy.io.loadmat
mat_sano = scipy.io.loadmat('ecg_sano.mat')
mat_taqui = scipy.io.loadmat('ecg_taqui.mat')

# Extrae el vector de datos (clave 'ecg' dentro de la estructura .mat)
data_sano = mat_sano.get('ecg', mat_sano.get('data'))
data_taqui = mat_taqui.get('ecg', mat_taqui.get('data'))

# Instanciar objetos de la clase Signal
paciente_sano = Signal(data=data_sano, label="Sano", fs=500)
paciente_taqui = Signal(data=data_taqui, label="Taquicardia", fs=500)

# ------------------------------------------------------------------------------
# 2.3 FILTRADO, RECORTE Y CÁLCULO DE ESTADÍSTICAS
# ------------------------------------------------------------------------------

# Rango de muestras a analizar 
inicio_muestras, fin_muestras = 0, 1000

# Paciente Sano
data_sano_filtrada = paciente_sano.filtrar_notch()
recorte_sano_orig = paciente_sano.recortar(canal=0, inicio=inicio_muestras, fin=fin_muestras)
recorte_sano_filt = paciente_sano.filtrar_notch()[:, inicio_muestras:fin_muestras][0]

media_sano, std_sano = paciente_sano.calcular_estadisticas(recorte_sano_filt)

# Paciente con Taquicardia
data_taqui_filtrada = paciente_taqui.filtrar_notch()
recorte_taqui_orig = paciente_taqui.recortar(canal=0, inicio=inicio_muestras, fin=fin_muestras)
recorte_taqui_filt = paciente_taqui.filtrar_notch()[:, inicio_muestras:fin_muestras][0]

media_taqui, std_taqui = paciente_taqui.calcular_estadisticas(recorte_taqui_filt)

# ------------------------------------------------------------------------------
# 2.4 RESULTADOS Y COMPARACIÓN ESTADÍSTICA
# ------------------------------------------------------------------------------

print("\n=== 2.5 COMPARACIÓN ESTADÍSTICA DE LAS SEÑALES ECG ===")
print(f"PACIENTE SANO:")
print(f"  - Media de amplitud: {media_sano:.4f} mV")
print(f"  - Desviación estándar: {std_sano:.4f} mV")

print(f"\nPACIENTE CON TAQUICARDIA:")
print(f"  - Media de amplitud: {media_taqui:.4f} mV")
print(f"  - Desviación estándar: {std_taqui:.4f} mV")

# ------------------------------------------------------------------------------
# 2.5 GRAFICACIÓN DE LOS SEGMENTOS PROCESADOS
# ------------------------------------------------------------------------------

paciente_sano.graficar_segmento(recorte_sano_orig, recorte_sano_filt, canal=0, inicio=inicio_muestras, fin=fin_muestras)
paciente_taqui.graficar_segmento(recorte_taqui_orig, recorte_taqui_filt, canal=0, inicio=inicio_muestras, fin=fin_muestras)

# ------------------------------------------------------------------------------
# 3. MANIPULACIÓN DE IMÁGENES CON OPENCV
# ------------------------------------------------------------------------------

import os
import cv2
import nibabel as nib


archivo_imagen = 'imagen_resonancia.nii'
if not os.path.exists(archivo_imagen):
    archivo_imagen = 'imagen_resonancia.nii.gz'
imagen_nii = nib.load(archivo_imagen)
volumen = imagen_nii.get_fdata()   

print("\n=== 3.1 INFORMACIÓN DE LA IMAGEN ===")
print(f"Dimensiones del volumen: {volumen.shape}")


indice_corte = volumen.shape[2] // 2
corte = volumen[:, :, indice_corte]

# grar el corte 90 grados para verlo en la orientación habitual
corte = np.rot90(corte)

if corte.max() > 255 or corte.min() < 0 or corte.dtype != np.uint8:
    corte = cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX)
img = corte.astype(np.uint8)

# ------------------------------------------------------------------------------
# 3.2 TRANSFORMACIONES
# ------------------------------------------------------------------------------

# 1) Filtrado Gaussiano
# Propósito biomédico: suaviza la imagen y reduce el ruido de la adquisición,
# aunque también hace los bordes un poco menos nítidos.
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 2) CLAHE (ecualización adaptativa del histograma)
# Propósito biomédico: mejora el contraste local, así se distinguen mejor
# estructuras con intensidades parecidas (materia gris, materia blanca, etc.).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img)

# 3) Umbralización adaptativa
# Propósito biomédico: separa regiones (tejido vs. fondo) usando un umbral
# calculado en cada vecindad, lo que sirve como paso previo de segmentación.
threshold = cv2.adaptiveThreshold(img, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

# 4) Detección de bordes (Canny)
# Propósito biomédico: resalta los contornos de las estructuras cerebrales,
# útil para delimitar formas y medir regiones.
edges = cv2.Canny(img, 100, 200)

# 5) Operación morfológica (apertura)
# Propósito biomédico: elimina puntos pequeños aislados (ruido) de la imagen
# binaria, dejando solo las regiones más grandes.
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

# 6) Transformación geométrica (rotación de 15 grados)
# Propósito biomédico: corrige la orientación o alinea la imagen con otra
# (registro de imágenes), por ejemplo si el paciente se movió.
filas, columnas = img.shape
M = cv2.getRotationMatrix2D((columnas / 2, filas / 2), 15, 1)
rotated = cv2.warpAffine(img, M, (columnas, filas))

# ------------------------------------------------------------------------------
# 3.3 MOSTRAR LAS IMÁGENES EN UNA SECUENCIA ORGANIZADA
# ------------------------------------------------------------------------------

imagenes = [img, blurred, enhanced, threshold, edges, opened, rotated]
titulos = ['Original', 'Filtro Gaussiano', 'CLAHE', 'Umbral adaptativo',
           'Bordes (Canny)', 'Apertura morfológica', 'Rotación 15°']

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.ravel()   # convierte la cuadricula de ejes en una lista simple

for i in range(len(imagenes)):
    axs[i].imshow(imagenes[i], cmap='gray')
    axs[i].set_title(titulos[i])
    axs[i].axis('off')

axs[7].axis('off')  

plt.suptitle('Transformaciones sobre un corte de resonancia magnética')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4. CONCLUSIÓN FINAL
# ------------------------------------------------------------------------------

print("\n=== 4.1 PROMEDIOS CLÍNICOS POR DIAGNÓSTICO ===")
promedios = df_clinico.groupby('Diagnóstico')[cols_numericas].mean()
print(promedios.round(2))

#================================================================================

# Diferencias observadas entre las señales ECG (sano vs taquicardia):


# - Del filtrado de observa que los dos primeros segundos la media de amplitud fue 
# 0.0529 mV en el paciente sano y 0.1004 mV en el de taquicardia.

# - La desviación estandar fue 0.1338 mV (sano) y 0.1769 mV (taquicardia).

# - En las graficas  el sano tiene un pico R (el latido) aproximadamente cada
#0.7 s, mientras que en la taquicardia el pico aparece cada 0.4 s. Eso son
#unos 86 latidos por minuto contra unos 150 latidos por minuto.

# - La media y la desviación son mayores en la taquicardia porque en el mismo
#recorte caben más picos R. No es que cada latido sea más
#grande, sino que hay más latidos en el mismo tiempo.

# - El filtro Notch de 60 Hz quitó la oscilación rápida del ruido eléctrico sin
#cambiar la forma de los latidos. Solo se nota una pequeña oscilación al
#inicio de la señal, que es un efecto de borde del filtro.

# Posibles relaciones entre los datos clínicos y las variables derivadas:


# - La frecuencia cardíaca promedio fue 70.06 LPM en los sanos y 123.62 LPM en
#los pacientes con taquicardia. Va en lamisma dirección que el ECG -la
#taquicardia tiene más latidos por minuto-, aunque los números no son iguales
#porque todos los datos clínicos son simulados y no corresponden a los pacientes de los ECG.


# - La PAM fue mas baja en taquicardia (82.78 contra 95.80 en sanos) y la
#glucosa mas alta (106.52 contra 88.60 en sanos).

# - La edad promedio fue 48.26 años (sanos) y 55.18 años (taquicardia), pero la
#desviacion estandar de cada grupo es de casi 19 años asi que la diferencia
#no es clara.


# Efecto de los filtros y transformaciones aplicadas a la imagen:


# - Gaussiano: la imagen se ve más borrosa, reduce el ruido fino, pero se pierde
#   nitidez en los surcos del cerebro
# - CLAHE: aumenta el contraste local y deja ver mejor los surcos y la
#   diferenca entre materia gris y materia blanca
# - Umbral adaptativo: produce una imagen en blanco y negro que marca los
#   contornos de los surcos, pero es muy sensible al ruido (aparecen muchos
#   detalles sueltos)
# - Canny: deja solo los bordes por lo que se ve bien el contorno del cráneo y de las
#   estructuras grandes, pero se pierde el detalle fino del interior
# - Apertura: elimina los detalles blancos peqeños del umbral, pero con un
#   kernel de 3x3 también rompe estructuras delgadas y deja la imagen más
#   cuadriculada
# - Rotación de 15°: el contenido no cambia solo la orientación de la imagen

#NOTA: mas arriba en el codigo le he pedido gemini que me explicara el proposito biomedico de cada transformacion
#(lo he dejado comentado)