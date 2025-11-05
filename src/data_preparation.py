"""
Módulo de Preparación de Datos - Versión Corregida
===================================================
Este módulo carga los datasets de energía y clima, los une y agrega por día.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data(energy_path='data/raw/energy_dataset.csv',
                          weather_path='data/raw/weather_features.csv'):
    """
    Carga los datasets, los une y prepara los datos agregados por día.
    
    Args:
        energy_path: Ruta al archivo CSV de energía
        weather_path: Ruta al archivo CSV de clima
        
    Returns:
        pd.DataFrame: DataFrame con datos diarios de temperatura y consumo
    """
    
    print("="*60)
    print("PREPARACIÓN DE DATOS")
    print("="*60)
    
    # 1. CARGAR DATASETS
    print("\n1. Cargando datasets...")
    
    # Cargar dataset de energía
    df_energy = pd.read_csv(energy_path)
    print(f"   - Dataset energía cargado: {df_energy.shape[0]} filas, {df_energy.shape[1]} columnas")
    
    # Cargar dataset de clima
    df_weather = pd.read_csv(weather_path)
    print(f"   - Dataset clima cargado: {df_weather.shape[0]} filas, {df_weather.shape[1]} columnas")
    
    # Ver qué ciudades hay disponibles
    cities = df_weather['city_name'].unique()
    print(f"   - Ciudades disponibles: {cities}")
    
    
    # 2. CONVERTIR COLUMNAS DE FECHA A DATETIME
    print("\n2. Convirtiendo fechas a datetime...")
    
    # Convertir la columna 'time' del dataset de energía
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)
    
    # Convertir la columna 'dt_iso' del dataset de clima  
    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    
    # Mostrar rango de fechas
    print(f"   - Rango energía: {df_energy['time'].min()} a {df_energy['time'].max()}")
    print(f"   - Rango clima: {df_weather['time'].min()} a {df_weather['time'].max()}")
    
    
    # 3. PREPARAR DATOS DE CLIMA (seleccionar una ciudad)
    print("\n3. Preparando datos de clima...")
    
    # Vamos a usar Madrid como referencia (o Barcelona si no hay Madrid)
    city_to_use = 'Madrid'
    if city_to_use not in cities:
        city_to_use = cities[0]  # Usar la primera ciudad disponible
        print(f"   - Madrid no disponible, usando {city_to_use}")
    else:
        print(f"   - Usando datos de {city_to_use}")
    
    # Filtrar por la ciudad seleccionada
    df_weather_city = df_weather[df_weather['city_name'] == city_to_use].copy()
    print(f"   - Registros para {city_to_use}: {len(df_weather_city)}")
    
    # Seleccionar solo las columnas necesarias
    df_weather_city = df_weather_city[['time', 'temp', 'temp_min', 'temp_max', 
                                       'humidity', 'pressure', 'wind_speed']].copy()
    
    
    # 4. UNIR LOS DOS DATASETS
    print("\n4. Uniendo datasets...")
    
    # Unir por la columna time
    df_combined = pd.merge(df_energy, df_weather_city, on='time', how='inner')
    print(f"   - Dataset combinado: {df_combined.shape[0]} filas")
    
    
    # 5. EXTRAER COLUMNAS RELEVANTES
    print("\n5. Seleccionando columnas relevantes...")
    
    # Para el consumo: usar 'total load actual' (demanda total real)
    # Para la temperatura: usar 'temp' (temperatura)
    
    # Verificar que las columnas existen
    if 'total load actual' not in df_combined.columns:
        print("   ⚠️ Columna 'total load actual' no encontrada")
        print(f"   Columnas disponibles: {df_combined.columns.tolist()[:10]}...")
    
    # La temperatura en el dataset de weather está en Kelvin, convertir a Celsius
    df_combined['temp_celsius'] = df_combined['temp'] - 273.15
    
    # Seleccionar solo las columnas necesarias
    df_selected = df_combined[['time', 'total load actual', 'temp_celsius']].copy()
    
    # Renombrar para claridad
    df_selected = df_selected.rename(columns={
        'total load actual': 'consumption',
        'temp_celsius': 'temperature'
    })
    
    
    # 6. AGREGAR POR DÍA
    print("\n6. Agregando datos por día...")
    
    # Crear columna de fecha (sin hora)
    df_selected['date'] = df_selected['time'].dt.date
    
    # Agrupar por día y calcular:
    # - Suma del consumo diario (MWh totales en el día)
    # - Media de la temperatura diaria (°C promedio del día)
    daily_data = df_selected.groupby('date').agg({
        'consumption': 'sum',      # Suma total del consumo en el día
        'temperature': 'mean'       # Temperatura media del día
    }).reset_index()
    
    print(f"   - Datos agregados: {daily_data.shape[0]} días")
    
    
    # 7. LIMPIAR DATOS
    print("\n7. Limpiando datos...")
    
    # Verificar valores nulos
    nulls_before = daily_data.isnull().sum()
    print(f"   - Valores nulos antes de limpiar:")
    for col in nulls_before.index:
        if nulls_before[col] > 0:
            print(f"     * {col}: {nulls_before[col]}")
    
    # Eliminar filas con valores nulos
    daily_data_clean = daily_data.dropna()
    
    rows_removed = len(daily_data) - len(daily_data_clean)
    if rows_removed > 0:
        print(f"   - Filas eliminadas: {rows_removed}")
    
    print(f"   - Dataset final: {daily_data_clean.shape[0]} días")
    
    
    # 8. MOSTRAR ESTADÍSTICAS
    print("\n8. Estadísticas del dataset final:")
    print("-"*40)
    print(daily_data_clean[['temperature', 'consumption']].describe())
    
    
    # 9. GUARDAR DATOS PROCESADOS
    print("\n9. Guardando datos procesados...")
    
    # Crear directorio si no existe
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar CSV
    output_path = output_dir / 'daily_consumption.csv'
    daily_data_clean.to_csv(output_path, index=False)
    print(f"   - Datos guardados en: {output_path}")
    
    return daily_data_clean


def main():
    """
    Función principal para ejecutar la preparación de datos.
    """
    # Ejecutar preparación
    df = load_and_prepare_data()
    
    print("\n" + "="*60)
    print("✅ PREPARACIÓN COMPLETADA")
    print("="*60)
    print(f"\nDataset listo para análisis:")
    print(f"  - Días: {len(df)}")
    print(f"  - Columnas: {df.columns.tolist()}")
    print(f"  - Rango de fechas: {df['date'].min()} a {df['date'].max()}")
    
    return df


if __name__ == "__main__":
    # Ejecutar si se llama directamente
    main()