"""
Script Principal - Análisis de Temperatura y Consumo Energético
================================================================
Paso 1: Preparación de datos (solo si es necesario)
Paso 2: Exploración inicial de datos
Paso 3: Entrenamiento y evaluación del modelo de regresión lineal
Paso 4: Visualización de resultados y análisis de ajuste
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Añadir la carpeta actual al path de Python para poder importar src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar la función de preparación de datos
from src.data_preparation import load_and_prepare_data


def cargar_o_preparar_datos():
    """
    Carga los datos procesados si existen, si no, los prepara.
    
    Returns:
        pd.DataFrame: DataFrame con los datos diarios
    """
    # Ruta del archivo procesado
    processed_file = Path('data/processed/daily_consumption.csv')
    
    # Verificar si el archivo ya existe
    if processed_file.exists():
        print("\n📂 Archivo de datos procesados encontrado!")
        print(f"   Cargando desde: {processed_file}")
        
        # Cargar datos existentes
        df_daily = pd.read_csv(processed_file)
        
        # Convertir columna date a datetime si es necesario
        if 'date' in df_daily.columns:
            df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        print(f"   ✅ Datos cargados: {len(df_daily)} días")
        print(f"   ✅ Columnas: {df_daily.columns.tolist()}")
        
    else:
        print("\n⚠️ Archivo de datos procesados no encontrado")
        print("   Ejecutando preparación de datos...")
        
        # Preparar datos desde cero
        df_daily = load_and_prepare_data()
        
        print("\n✅ Datos preparados y guardados!")
    
    return df_daily


def exploracion_inicial(df):
    """
    Realiza la exploración inicial de los datos.
    
    Args:
        df: DataFrame con los datos diarios de temperatura y consumo
    """
    print("\n" + "="*60)
    print(" PASO 2: EXPLORACIÓN INICIAL DE DATOS")
    print("="*60)
    
    # 1. Información básica del DataFrame
    print("\n1️⃣ Información del DataFrame:")
    print("-"*40)
    print(f"   - Número de filas: {len(df)}")
    print(f"   - Número de columnas: {len(df.columns)}")
    print(f"   - Columnas: {df.columns.tolist()}")
    print(f"   - Tipos de datos:")
    for col in df.columns:
        print(f"     * {col}: {df[col].dtype}")
    
    # 2. Primeras 5 filas
    print("\n2️⃣ Primeras 5 filas del dataset:")
    print("-"*40)
    print(df.head())
    
    # 3. Estadísticas descriptivas
    print("\n3️⃣ Estadísticas descriptivas:")
    print("-"*40)
    print(df[['temperature', 'consumption']].describe())
    
    # 4. Valores nulos
    print("\n4️⃣ Verificación de valores nulos:")
    print("-"*40)
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("   ✅ No hay valores nulos en el dataset")
    else:
        print("   ⚠️ Valores nulos encontrados:")
        for col in nulls.index:
            if nulls[col] > 0:
                print(f"     * {col}: {nulls[col]}")
    
    # 5. Gráfico de dispersión
    print("\n5️⃣ Creando gráfico de dispersión inicial...")
    print("-"*40)
    
    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Crear gráfico de dispersión
    scatter = ax.scatter(df['temperature'], 
                        df['consumption'], 
                        alpha=0.5,
                        s=30,
                        c='blue',
                        edgecolors='black',
                        linewidth=0.5)
    
    # Configurar etiquetas y título
    ax.set_xlabel('Temperatura (°C)', fontsize=12)
    ax.set_ylabel('Consumo Eléctrico (MWh)', fontsize=12)
    ax.set_title('Relación entre Temperatura y Consumo Eléctrico', 
                fontsize=14, fontweight='bold')
    
    # Añadir grilla
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir estadísticas en el gráfico
    correlation = df['temperature'].corr(df['consumption'])
    textstr = f'Correlación: {correlation:.3f}\nN = {len(df)} días'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/scatter_temperatura_consumo.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico guardado en: {output_path}")
    
    # Mostrar el gráfico
    plt.show()
    
    print("\n✅ Exploración inicial completada!")


def entrenar_modelo_regresion(df):
    """
    Entrena un modelo de regresión lineal y evalúa su rendimiento.
    
    Args:
        df: DataFrame con los datos de temperatura y consumo
        
    Returns:
        tuple: (modelo, X, y, metricas)
    """
    print("\n" + "="*60)
    print(" PASO 3: ENTRENAMIENTO Y EVALUACIÓN DEL MODELO")
    print("="*60)
    
    # 1. Preparar datos para el modelo
    print("\n1️⃣ Preparando datos para el modelo:")
    print("-"*40)
    
    # Variables independiente (X) y dependiente (y)
    X = df[['temperature']].values  # Necesita ser 2D para sklearn
    y = df['consumption'].values
    
    print(f"   - Variable independiente (X): temperatura")
    print(f"   - Variable dependiente (y): consumo")
    print(f"   - Tamaño del dataset: {len(X)} observaciones")
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   - Conjunto de entrenamiento: {len(X_train)} observaciones (80%)")
    print(f"   - Conjunto de prueba: {len(X_test)} observaciones (20%)")
    
    # 2. Entrenar el modelo
    print("\n2️⃣ Entrenando modelo de regresión lineal:")
    print("-"*40)
    
    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Obtener parámetros del modelo
    slope = model.coef_[0]  # Pendiente (w)
    intercept = model.intercept_  # Intercepto (b)
    
    print(f"   ✅ Modelo entrenado!")
    print(f"   - Ecuación: consumo = {slope:.2f} × temperatura + {intercept:.2f}")
    print(f"   - Pendiente (w): {slope:.2f}")
    print(f"   - Intercepto (b): {intercept:.2f}")
    
    # 3. Hacer predicciones
    print("\n3️⃣ Realizando predicciones:")
    print("-"*40)
    
    # Predicciones en conjunto de prueba
    y_pred_test = model.predict(X_test)
    
    # Predicciones en todo el dataset (para visualización)
    y_pred = model.predict(X)
    
    print(f"   ✅ Predicciones realizadas en {len(X_test)} muestras de prueba")
    
    # 4. Calcular métricas
    print("\n4️⃣ Evaluación del modelo:")
    print("-"*40)
    
    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    # R² (Coeficiente de determinación)
    r2 = r2_score(y_test, y_pred_test)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_test - y_pred_test))
    
    print(f"\n   📊 MÉTRICAS DE EVALUACIÓN:")
    print(f"   {'='*35}")
    print(f"   - MSE (Error Cuadrático Medio): {mse:.2f}")
    print(f"   - RMSE (Raíz del MSE): {rmse:.2f} MWh")
    print(f"   - MAE (Error Absoluto Medio): {mae:.2f} MWh")
    print(f"   - R² (Coef. de Determinación): {r2:.4f}")
    
    # 5. Interpretación de las métricas
    print("\n5️⃣ Interpretación de las métricas:")
    print("-"*40)
    
    print(f"\n   📈 MSE = {mse:.2f}")
    print("      → Promedio de los errores al cuadrado")
    print("      → Penaliza más los errores grandes")
    print(f"      → En promedio, el error al cuadrado es {mse:.2f} MWh²")
    
    print(f"\n   📈 RMSE = {rmse:.2f} MWh")
    print("      → Error típico en las mismas unidades que el consumo")
    print(f"      → Las predicciones se desvían ±{rmse:.2f} MWh en promedio")
    
    print(f"\n   📈 R² = {r2:.4f} ({r2*100:.2f}%)")
    print(f"      → El modelo explica el {r2*100:.2f}% de la variabilidad del consumo")
    
    if r2 < 0.3:
        print("      ⚠️ Ajuste POBRE: El modelo no captura bien la relación")
        print("      💡 Posible relación no lineal o faltan variables")
    elif r2 < 0.7:
        print("      📊 Ajuste MODERADO: Hay margen de mejora")
        print("      💡 Considerar modelos más complejos o más variables")
    else:
        print("      ✅ Ajuste BUENO: El modelo explica bien la variabilidad")
    
    # Retornar resultados
    metricas = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'intercept': intercept
    }
    
    print("\n✅ Entrenamiento y evaluación completados!")
    
    return model, X, y, metricas


def visualizar_resultados(df, model, X, y, metricas):
    """
    PASO 4: Visualiza los resultados del modelo y analiza el ajuste.
    
    Args:
        df: DataFrame original
        model: Modelo entrenado
        X: Variables independientes
        y: Variable dependiente
        metricas: Diccionario con métricas del modelo
    """
    print("\n" + "="*60)
    print(" PASO 4: VISUALIZACIÓN DE RESULTADOS")
    print("="*60)
    
    print("\n1️⃣ Creando visualización del modelo ajustado...")
    print("-"*40)
    
    # Predicciones del modelo
    y_pred = model.predict(X)
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # ========== SUBPLOT 1: Dispersión + Recta de Regresión ==========
    ax1 = plt.subplot(1, 3, 1)
    
    # Scatter plot de los datos
    ax1.scatter(X, y, alpha=0.6, s=30, color='navy', 
               edgecolors='black', linewidth=0.5, label='Datos reales')
    
    # Línea de regresión
    ax1.plot(X, y_pred, 'r-', linewidth=2.5, 
            label=f'Regresión lineal (R²={metricas["r2"]:.3f})')
    
    # Configuración
    ax1.set_xlabel('Temperatura (°C)', fontsize=12)
    ax1.set_ylabel('Consumo (MWh)', fontsize=12)
    ax1.set_title('Modelo de Regresión Lineal Ajustado', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir ecuación
    equation = f'y = {metricas["slope"]:.2f}x + {metricas["intercept"]:.2f}'
    ax1.text(0.05, 0.95, equation, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # ========== SUBPLOT 2: Análisis de Residuos ==========
    ax2 = plt.subplot(1, 3, 2)
    
    # Calcular residuos
    residuos = y - y_pred
    
    # Scatter de residuos
    ax2.scatter(X, residuos, alpha=0.6, s=30, color='green',
               edgecolors='black', linewidth=0.5)
    
    # Línea horizontal en y=0
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Configuración
    ax2.set_xlabel('Temperatura (°C)', fontsize=12)
    ax2.set_ylabel('Residuos (MWh)', fontsize=12)
    ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir estadísticas de residuos
    residuo_std = np.std(residuos)
    ax2.text(0.05, 0.95, f'Std residuos: {residuo_std:.2f}', 
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # ========== SUBPLOT 3: Análisis de Curvatura ==========
    ax3 = plt.subplot(1, 3, 3)
    
    # Agrupar datos por rangos de temperatura para ver tendencia
    temp_bins = pd.cut(df['temperature'], bins=10)
    grouped = df.groupby(temp_bins).agg({
        'temperature': 'mean',
        'consumption': 'mean'
    }).reset_index(drop=True)
    
    # Plot de medias por bin
    ax3.scatter(grouped['temperature'], grouped['consumption'], 
               s=100, color='orange', edgecolors='black', linewidth=1,
               label='Medias por rango', zorder=3)
    
    # Línea que une los puntos para ver curvatura
    ax3.plot(grouped['temperature'], grouped['consumption'], 
            'b--', linewidth=1.5, alpha=0.7, label='Tendencia observada')
    
    # Predicción del modelo lineal sobre las medias
    X_grouped = grouped[['temperature']].values
    y_pred_grouped = model.predict(X_grouped)
    ax3.plot(grouped['temperature'], y_pred_grouped, 
            'r-', linewidth=2, label='Modelo lineal')
    
    # Configuración
    ax3.set_xlabel('Temperatura (°C)', fontsize=12)
    ax3.set_ylabel('Consumo Medio (MWh)', fontsize=12)
    ax3.set_title('Análisis de Linealidad', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Guardar figura
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/analisis_completo_modelo.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico guardado en: {output_path}")
    
    plt.show()
    
    # ========== ANÁLISIS DE AJUSTE ==========
    print("\n2️⃣ Análisis del ajuste del modelo:")
    print("-"*40)
    
    # Analizar patrón en residuos
    print("\n   📊 ANÁLISIS DE RESIDUOS:")
    
    # Calcular correlación entre temperatura y residuos
    corr_temp_residuos = np.corrcoef(X.flatten(), residuos)[0, 1]
    
    if abs(corr_temp_residuos) < 0.1:
        print(f"   ✅ Residuos aleatorios (correlación = {corr_temp_residuos:.3f})")
        print("      → No hay patrón evidente en los residuos")
    else:
        print(f"   ⚠️ Posible patrón en residuos (correlación = {corr_temp_residuos:.3f})")
        print("      → Los residuos muestran tendencia sistemática")
    
    # Analizar curvatura
    print("\n   📊 ANÁLISIS DE CURVATURA:")
    
    # Verificar si hay forma de U (consumo alto en extremos)
    temp_low = df[df['temperature'] < df['temperature'].quantile(0.25)]['consumption'].mean()
    temp_mid = df[(df['temperature'] >= df['temperature'].quantile(0.25)) & 
                  (df['temperature'] <= df['temperature'].quantile(0.75))]['consumption'].mean()
    temp_high = df[df['temperature'] > df['temperature'].quantile(0.75)]['consumption'].mean()
    
    if temp_low > temp_mid and temp_high > temp_mid:
        print("   ⚠️ RELACIÓN EN FORMA DE U DETECTADA")
        print("      → Consumo alto en temperaturas extremas (frío y calor)")
        print("      → Un modelo polinómico podría ajustar mejor")
        print(f"      - Consumo temp. bajas: {temp_low:.2f} MWh")
        print(f"      - Consumo temp. medias: {temp_mid:.2f} MWh")
        print(f"      - Consumo temp. altas: {temp_high:.2f} MWh")
    elif metricas['slope'] < 0:
        print("   📉 RELACIÓN LINEAL NEGATIVA")
        print("      → El consumo disminuye al aumentar la temperatura")
        print("      → Modelo lineal es apropiado para esta tendencia")
    else:
        print("   📈 RELACIÓN LINEAL POSITIVA")
        print("      → El consumo aumenta con la temperatura")
        print("      → Modelo lineal captura la tendencia general")
    
    # Recomendaciones finales
    print("\n3️⃣ Conclusiones sobre el ajuste:")
    print("-"*40)
    
    if metricas['r2'] < 0.3:
        print("   ⚠️ El modelo lineal NO se ajusta bien a los datos")
        print("   💡 Recomendaciones:")
        print("      1. Probar regresión polinómica (grado 2 o 3)")
        print("      2. Considerar transformación logarítmica")
        print("      3. Añadir más variables predictoras")
    elif metricas['r2'] < 0.7:
        print("   📊 El modelo lineal tiene un ajuste MODERADO")
        print("   💡 Recomendaciones:")
        print("      1. Explorar modelos no lineales para mejorar")
        print("      2. Analizar por segmentos de temperatura")
        print("      3. Incluir variables como día de la semana")
    else:
        print("   ✅ El modelo lineal se ajusta BIEN a los datos")
        print("   💡 El modelo captura adecuadamente la relación")
    
    print("\n✅ Visualización y análisis completados!")


def main():
    """
    Función principal del proyecto.
    """
    
    print("\n" + "="*60)
    print(" ANÁLISIS DE TEMPERATURA Y CONSUMO ENERGÉTICO")
    print("="*60)
    
    # PASO 1: Cargar o preparar los datos
    print("\n📊 PASO 1: Carga/Preparación de datos")
    print("-"*40)
    
    try:
        df_daily = cargar_o_preparar_datos()
        
    except Exception as e:
        print(f"\n❌ Error durante la carga/preparación: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # PASO 2: Exploración inicial
    try:
        exploracion_inicial(df_daily)
        
    except Exception as e:
        print(f"\n❌ Error durante la exploración: {e}")
        import traceback
        traceback.print_exc()
    
    # PASO 3: Entrenamiento y evaluación del modelo
    try:
        model, X, y, metricas = entrenar_modelo_regresion(df_daily)
        
    except Exception as e:
        print(f"\n❌ Error durante el modelado: {e}")
        import traceback
        traceback.print_exc()
        return df_daily
    
    # PASO 4: Visualización de resultados
    try:
        visualizar_resultados(df_daily, model, X, y, metricas)
        
    except Exception as e:
        print(f"\n❌ Error durante la visualización: {e}")
        import traceback
        traceback.print_exc()
        
    return df_daily
    

if __name__ == "__main__":
    # Ejecutar el programa
    df = main()
    
    if df is not None:
        print("\n" + "="*60)
        print(" ✅ ANÁLISIS COMPLETADO - TODOS LOS PASOS")
        print("="*60)
        
        # Resumen final
        print("\n📊 RESUMEN FINAL DEL PROYECTO:")
        print("-"*40)
        print(f"  ✓ Total de días analizados: {len(df)}")
        print(f"  ✓ Temperatura media: {df['temperature'].mean():.2f}°C")
        print(f"  ✓ Consumo medio: {df['consumption'].mean():.2f} MWh")
        print(f"  ✓ Correlación temperatura-consumo: {df['temperature'].corr(df['consumption']):.3f}")
        print("\n  📁 Resultados guardados en:")
        print("     - Datos: data/processed/")
        print("     - Gráficos: results/figures/")