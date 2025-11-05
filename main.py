"""
Script Principal - Análisis de Temperatura y Consumo Energético
================================================================
Paso 1: Preparación de datos (solo si es necesario)
Paso 2: Exploración inicial de datos
Paso 3: Entrenamiento y evaluación del modelo de regresión lineal
Paso 4: Visualización de resultados y análisis de ajuste
Paso 5: Análisis con subsets de temperatura
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
from sklearn.preprocessing import PolynomialFeatures

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
    
    print("\n✅ Visualización completada!")


def analizar_subsets(df_full):
    """
    PASO 5: Analiza el modelo en diferentes subsets de temperatura.
    
    Args:
        df_full: DataFrame completo con todos los datos
    """
    print("\n" + "="*60)
    print(" PASO 5: ANÁLISIS CON SUBSETS DE TEMPERATURA")
    print("="*60)
    
    print("\n📊 Objetivo: Encontrar rangos donde el modelo lineal funcione mejor")
    print("-"*60)
    
    # Definir los 3 subsets de temperatura
    subsets = [
        {'name': 'Temperaturas Moderadas', 'min': 10, 'max': 25, 'color': 'green'},
        {'name': 'Temperaturas Frías', 'min': 0, 'max': 15, 'color': 'blue'},
        {'name': 'Temperaturas Cálidas', 'min': 20, 'max': 35, 'color': 'red'}
    ]
    
    # Lista para guardar resultados
    results = []
    
    # Analizar cada subset
    for subset in subsets:
        print(f"\n{'='*50}")
        print(f" Analizando: {subset['name']} ({subset['min']}°C - {subset['max']}°C)")
        print('='*50)
        
        # Filtrar datos
        df_subset = df_full[
            (df_full['temperature'] >= subset['min']) & 
            (df_full['temperature'] <= subset['max'])
        ].copy()
        
        print(f"   📈 Datos en el subset: {len(df_subset)} días ({len(df_subset)/len(df_full)*100:.1f}%)")
        
        if len(df_subset) < 20:
            print("   ⚠️ Muy pocos datos para este subset, saltando...")
            continue
        
        # Preparar datos
        X = df_subset[['temperature']].values
        y = df_subset['consumption'].values
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_test = model.predict(X_test)
        y_pred = model.predict(X)
        
        # Calcular métricas
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Guardar resultados
        results.append({
            'name': subset['name'],
            'range': f"{subset['min']}-{subset['max']}°C",
            'n_samples': len(df_subset),
            'r2': r2,
            'rmse': rmse,
            'slope': slope,
            'intercept': intercept,
            'X': X,
            'y': y,
            'y_pred': y_pred,
            'color': subset['color'],
            'model': model
        })
        
        # Mostrar resultados
        print(f"\n   📊 RESULTADOS DEL SUBSET:")
        print(f"   - Ecuación: y = {slope:.2f}x + {intercept:.2f}")
        print(f"   - R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"   - RMSE: {rmse:.2f} MWh")
        
        if r2 > 0.5:
            print(f"   ✅ Buen ajuste en este rango!")
        else:
            print(f"   ⚠️ Ajuste moderado/pobre en este rango")
    
    # ========== COMPARACIÓN VISUAL DE LOS 3 SUBSETS ==========
    print("\n" + "="*60)
    print(" COMPARACIÓN VISUAL DE SUBSETS")
    print("="*60)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Primera fila: Gráficos individuales de cada subset
    for i, result in enumerate(results, 1):
        ax = plt.subplot(2, 3, i)
        
        # Scatter plot
        ax.scatter(result['X'], result['y'], 
                  alpha=0.6, s=30, color=result['color'],
                  edgecolors='black', linewidth=0.5, label='Datos reales')
        
        # Línea de regresión
        ax.plot(result['X'], result['y_pred'], 
               color='darkred', linewidth=2.5,
               label=f"R²={result['r2']:.3f}")
        
        # Configuración
        ax.set_xlabel('Temperatura (°C)', fontsize=10)
        ax.set_ylabel('Consumo (MWh)', fontsize=10)
        ax.set_title(f"{result['name']}\n({result['range']})", 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Ecuación
        equation = f"y = {result['slope']:.1f}x + {result['intercept']:.0f}"
        ax.text(0.05, 0.95, equation, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Segunda fila: Comparación conjunta
    ax4 = plt.subplot(2, 3, 4)
    
    # Datos completos en gris
    ax4.scatter(df_full['temperature'], df_full['consumption'],
               alpha=0.2, s=20, color='gray', label='Todos los datos')
    
    # Superponer cada subset con su color
    for result in results:
        ax4.scatter(result['X'], result['y'],
                   alpha=0.6, s=30, color=result['color'],
                   label=result['name'])
    
    ax4.set_xlabel('Temperatura (°C)', fontsize=11)
    ax4.set_ylabel('Consumo (MWh)', fontsize=11)
    ax4.set_title('Todos los Subsets Superpuestos', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Gráfico de barras comparativo de R²
    ax5 = plt.subplot(2, 3, 5)
    
    names = [r['name'].replace('Temperaturas ', '') for r in results]
    r2_values = [r['r2'] for r in results]
    colors = [r['color'] for r in results]
    
    bars = ax5.bar(names, r2_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Añadir valores en las barras
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax5.set_ylabel('R² Score', fontsize=11)
    ax5.set_title('Comparación de R² por Subset', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Tabla resumen
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Crear tabla de comparación
    table_data = [
        ['Subset', 'N', 'R²', 'RMSE', 'Pendiente']
    ]
    
    for r in results:
        table_data.append([
            r['name'].replace('Temperaturas ', ''),
            str(r['n_samples']),
            f"{r['r2']:.3f}",
            f"{r['rmse']:.1f}",
            f"{r['slope']:.1f}"
        ])
    
    # Modelo completo para comparar
    X_full = df_full[['temperature']].values
    y_full = df_full['consumption'].values
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    model_full = LinearRegression()
    model_full.fit(X_train, y_train)
    y_pred_test_full = model_full.predict(X_test)
    r2_full = r2_score(y_test, y_pred_test_full)
    rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_test_full))
    
    table_data.append([
        'COMPLETO',
        str(len(df_full)),
        f"{r2_full:.3f}",
        f"{rmse_full:.1f}",
        f"{model_full.coef_[0]:.1f}"
    ])
    
    table = ax6.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Colorear encabezado
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear fila del dataset completo
    for i in range(5):
        table[(len(table_data)-1, i)].set_facecolor('#ffcccc')
    
    ax6.set_title('Tabla Comparativa', fontsize=12, fontweight='bold')
    
    plt.suptitle('ANÁLISIS POR SUBSETS DE TEMPERATURA', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Guardar figura
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/analisis_subsets_temperatura.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Gráfico comparativo guardado en: {output_path}")
    
    plt.show()
    
    # ========== CONCLUSIONES ==========
    print("\n" + "="*60)
    print(" CONCLUSIONES DEL ANÁLISIS POR SUBSETS")
    print("="*60)
    
    # Encontrar el mejor subset
    best_subset = max(results, key=lambda x: x['r2'])
    
    print(f"\n🏆 MEJOR SUBSET: {best_subset['name']} ({best_subset['range']})")
    print(f"   - R² = {best_subset['r2']:.4f} (vs {r2_full:.4f} del modelo completo)")
    print(f"   - Mejora del {((best_subset['r2']-r2_full)/r2_full*100):.1f}% respecto al modelo completo")
    
    print("\n📊 ANÁLISIS COMPARATIVO:")
    for result in results:
        mejora = ((result['r2']-r2_full)/r2_full*100)
        if mejora > 0:
            print(f"   ✅ {result['name']}: R²={result['r2']:.3f} (+{mejora:.1f}% mejora)")
        else:
            print(f"   ⚠️ {result['name']}: R²={result['r2']:.3f} ({mejora:.1f}% peor)")
    
    print("\n💡 INTERPRETACIÓN:")
    print("   - El modelo lineal funciona mejor en rangos específicos de temperatura")
    print("   - Esto sugiere que la relación completa NO es perfectamente lineal")
    print("   - En temperaturas extremas (muy frías o muy cálidas) hay comportamientos diferentes")
    print("   - Un modelo segmentado o polinómico podría capturar mejor la relación completa")
    
    print("\n✅ Análisis por subsets completado!")


def regresion_polinomica(df):
    """
    PASO EXTRA: Entrena y evalúa un modelo de regresión polinómica de grado 2.
    Compara con el modelo lineal para ver si captura mejor la relación en forma de U.
    
    Args:
        df: DataFrame con los datos de temperatura y consumo
    """
    print("\n" + "="*60)
    print(" PASO EXTRA: REGRESIÓN POLINÓMICA (GRADO 2)")
    print("="*60)
    
    from sklearn.preprocessing import PolynomialFeatures
    
    print("\n📊 Objetivo: Capturar la relación no lineal (forma de U) con un modelo polinómico")
    print("-"*60)
    
    # 1. Preparar datos
    print("\n1️⃣ Preparando datos para modelo polinómico...")
    
    X = df[['temperature']].values
    y = df['consumption'].values
    
    # Crear características polinómicas de grado 2
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    print(f"   - Características originales: 1 (temperatura)")
    print(f"   - Características polinómicas: {X_poly.shape[1]} (temperatura, temperatura²)")
    print(f"   - Tamaño del dataset: {len(X)} observaciones")
    
    # 2. Dividir en train/test
    X_train_poly, X_test_poly, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # También necesitamos los datos originales para comparación
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Entrenar modelo polinómico
    print("\n2️⃣ Entrenando modelo polinómico...")
    
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    
    # Obtener coeficientes
    coef_linear = model_poly.coef_[0]  # Coeficiente de x
    coef_quadratic = model_poly.coef_[1]  # Coeficiente de x²
    intercept = model_poly.intercept_
    
    print(f"   ✅ Modelo entrenado!")
    print(f"   - Ecuación: consumo = {coef_quadratic:.2f}×temp² + {coef_linear:.2f}×temp + {intercept:.2f}")
    
    # 4. Entrenar modelo lineal para comparación
    print("\n3️⃣ Entrenando modelo lineal para comparación...")
    
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    
    # 5. Predicciones
    y_pred_poly = model_poly.predict(X_test_poly)
    y_pred_linear = model_linear.predict(X_test)
    
    # Predicciones en todo el dataset para visualización
    y_pred_poly_all = model_poly.predict(X_poly)
    y_pred_linear_all = model_linear.predict(X)
    
    # 6. Calcular métricas
    print("\n4️⃣ Evaluación y comparación de modelos:")
    print("-"*40)
    
    # Métricas modelo polinómico
    r2_poly = r2_score(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    mae_poly = np.mean(np.abs(y_test - y_pred_poly))
    
    # Métricas modelo lineal
    r2_linear = r2_score(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
    mae_linear = np.mean(np.abs(y_test - y_pred_linear))
    
    print("\n   📊 COMPARACIÓN DE MODELOS:")
    print("   " + "="*50)
    print(f"   {'Métrica':<15} {'Lineal':<15} {'Polinómico':<15} {'Mejora':<15}")
    print("   " + "-"*50)
    print(f"   {'R²':<15} {r2_linear:.4f}{'':<9} {r2_poly:.4f}{'':<9} {'+' if r2_poly > r2_linear else ''}{abs(r2_poly - r2_linear):.4f}")
    print(f"   {'RMSE (MWh)':<15} {rmse_linear:.2f}{'':<9} {rmse_poly:.2f}{'':<9} {'-' if rmse_poly < rmse_linear else '+'}{abs(rmse_poly - rmse_linear):.2f}")
    print(f"   {'MAE (MWh)':<15} {mae_linear:.2f}{'':<9} {mae_poly:.2f}{'':<9} {'-' if mae_poly < mae_linear else '+'}{abs(mae_poly - mae_linear):.2f}")
    
    # Calcular mejora porcentual
    mejora_r2 = ((r2_poly - r2_linear) / abs(r2_linear)) * 100 if r2_linear != 0 else 0
    mejora_rmse = ((rmse_linear - rmse_poly) / rmse_linear) * 100
    
    print("\n   📈 MEJORA DEL MODELO POLINÓMICO:")
    print(f"   - R² mejoró en: {mejora_r2:.1f}%")
    print(f"   - RMSE mejoró en: {mejora_rmse:.1f}%")
    
    if r2_poly > r2_linear * 1.2:  # Si mejora más del 20%
        print("   ✅ El modelo polinómico es SIGNIFICATIVAMENTE mejor")
        print("   → Confirma relación no lineal (forma de U)")
    elif r2_poly > r2_linear:
        print("   📊 El modelo polinómico es ligeramente mejor")
    else:
        print("   ⚠️ El modelo polinómico no mejora significativamente")
    
    # 7. Visualización comparativa
    print("\n5️⃣ Creando visualización comparativa...")
    print("-"*40)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Modelo Lineal
    ax1 = axes[0]
    ax1.scatter(X, y, alpha=0.4, s=20, color='gray', label='Datos reales')
    
    # Ordenar para una línea suave
    idx_sort = X.flatten().argsort()
    X_sorted = X[idx_sort]
    y_pred_linear_sorted = y_pred_linear_all[idx_sort]
    
    ax1.plot(X_sorted, y_pred_linear_sorted, 'b-', linewidth=2.5, 
             label=f'Lineal (R²={r2_linear:.3f})')
    ax1.set_xlabel('Temperatura (°C)', fontsize=11)
    ax1.set_ylabel('Consumo (MWh)', fontsize=11)
    ax1.set_title('Modelo Lineal', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Modelo Polinómico
    ax2 = axes[1]
    ax2.scatter(X, y, alpha=0.4, s=20, color='gray', label='Datos reales')
    
    y_pred_poly_sorted = y_pred_poly_all[idx_sort]
    ax2.plot(X_sorted, y_pred_poly_sorted, 'r-', linewidth=2.5,
             label=f'Polinómico G2 (R²={r2_poly:.3f})')
    ax2.set_xlabel('Temperatura (°C)', fontsize=11)
    ax2.set_ylabel('Consumo (MWh)', fontsize=11)
    ax2.set_title('Modelo Polinómico (Grado 2)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Añadir ecuación
    if coef_quadratic >= 0:
        equation = f'y = {coef_quadratic:.1f}x² + {coef_linear:.1f}x + {intercept:.0f}'
    else:
        equation = f'y = {coef_quadratic:.1f}x² + {coef_linear:.1f}x + {intercept:.0f}'
    ax2.text(0.05, 0.95, equation, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Subplot 3: Comparación directa
    ax3 = axes[2]
    ax3.scatter(X, y, alpha=0.3, s=15, color='gray', label='Datos reales')
    ax3.plot(X_sorted, y_pred_linear_sorted, 'b-', linewidth=2, 
             label=f'Lineal (R²={r2_linear:.3f})', alpha=0.8)
    ax3.plot(X_sorted, y_pred_poly_sorted, 'r-', linewidth=2,
             label=f'Polinómico (R²={r2_poly:.3f})', alpha=0.8)
    ax3.set_xlabel('Temperatura (°C)', fontsize=11)
    ax3.set_ylabel('Consumo (MWh)', fontsize=11)
    ax3.set_title('Comparación de Modelos', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Añadir análisis de la forma
    if coef_quadratic > 0:
        vertex_x = -coef_linear / (2 * coef_quadratic)
        ax3.axvline(x=vertex_x, color='green', linestyle='--', alpha=0.5, 
                   label=f'Mínimo en {vertex_x:.1f}°C')
        ax3.legend()
        
    plt.suptitle('COMPARACIÓN: MODELO LINEAL vs POLINÓMICO', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Guardar figura
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/comparacion_lineal_vs_polinomico.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico guardado en: {output_path}")
    
    plt.show()
    
    # 8. Análisis de la curva
    print("\n6️⃣ Análisis de la curva polinómica:")
    print("-"*40)
    
    if coef_quadratic > 0:
        print("   📊 Parábola con forma de U (coeficiente cuadrático positivo)")
        print("   → Consumo alto en temperaturas extremas (frío y calor)")
        print("   → Consumo mínimo en temperaturas moderadas")
        vertex_x = -coef_linear / (2 * coef_quadratic)
        print(f"   → Temperatura óptima (mínimo consumo): {vertex_x:.1f}°C")
    else:
        print("   📊 Parábola invertida (coeficiente cuadrático negativo)")
        print("   → Patrón diferente al esperado")
    
    # 9. Conclusiones
    print("\n" + "="*60)
    print(" CONCLUSIONES DEL MODELO POLINÓMICO")
    print("="*60)
    
    print("\n✅ RESULTADOS CLAVE:")
    print(f"   1. El modelo polinómico {'MEJORA' if r2_poly > r2_linear else 'NO MEJORA'} el ajuste")
    print(f"   2. R² pasó de {r2_linear:.3f} (lineal) a {r2_poly:.3f} (polinómico)")
    print(f"   3. Esto representa una mejora del {mejora_r2:.1f}%")
    
    if r2_poly > 0.5:
        print("\n💡 INTERPRETACIÓN:")
        print("   - La relación temperatura-consumo NO es lineal")
        print("   - Hay un patrón en forma de U: alto consumo en extremos")
        print("   - El modelo polinómico captura mejor esta relación")
        print("   - En la práctica, esto refleja el uso de calefacción (frío) y aire acondicionado (calor)")
    else:
        print("\n💡 INTERPRETACIÓN:")
        print("   - Aunque el modelo polinómico mejora, el R² sigue siendo moderado")
        print("   - Esto sugiere que otros factores influyen en el consumo")
        print("   - Posibles factores: día de la semana, estacionalidad, eventos especiales")
    
    print("\n✅ Análisis polinómico completado!")
    
    return {
        'model_poly': model_poly,
        'model_linear': model_linear,
        'r2_poly': r2_poly,
        'r2_linear': r2_linear,
        'mejora_r2': mejora_r2
    }


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
    
    # PASO 5: Análisis con subsets
    try:
        analizar_subsets(df_daily)
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis de subsets: {e}")
        import traceback
        traceback.print_exc()

    # PASO EXTRA: Regresión polinómica
    try:
        resultados_poly = regresion_polinomica(df_daily)
    except Exception as e:
        print(f"\n❌ Error durante la regresión polinómica: {e}")
        import traceback
        traceback.print_exc()
        
    return df_daily
    

if __name__ == "__main__":
    # Ejecutar el programa
    df = main()
    
    if df is not None:
        print("\n" + "="*60)
        print(" ✅ PROYECTO COMPLETADO - TODOS LOS PASOS REALIZADOS")
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
        print("\n  🎯 Conclusión principal:")
        print("     El modelo lineal funciona mejor en rangos específicos de temperatura")
        print("     que en el dataset completo, sugiriendo una relación no lineal.")