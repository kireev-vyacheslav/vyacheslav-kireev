import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Размер train: {train.shape}, test: {test.shape}")

# Проверяем наличие row_id в тестовых данных
if 'row_id' not in test.columns:
    test['row_id'] = test.index + 1

# Функция для расширенного создания признаков
def create_advanced_features(df, train_df=None):
    """Создание расширенных признаков"""
    df = df.copy()
    
    # Преобразование даты
    if 'dt' in df.columns:
        df['dt'] = pd.to_datetime(df['dt'])
        df['year'] = df['dt'].dt.year
        df['quarter'] = df['dt'].dt.quarter
        df['day_of_year'] = df['dt'].dt.dayofyear
        df['is_month_start'] = df['dt'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['dt'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['dt'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['dt'].dt.is_quarter_end.astype(int)
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        
        # Циклические признаки для дня недели и месяца
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Сложные взаимодействия признаков
    if all(col in df.columns for col in ['avg_temperature', 'avg_humidity']):
        df['temp_humidity'] = df['avg_temperature'] * df['avg_humidity']
        df['temp_humidity_ratio'] = df['avg_temperature'] / (df['avg_humidity'] + 1e-6)
        df['comfort_index'] = 0.5 * df['avg_temperature'] + 0.3 * (100 - df['avg_humidity'])
    
    if all(col in df.columns for col in ['precpt', 'avg_wind_level']):
        df['weather_severity'] = df['precpt'] + abs(df['avg_wind_level'])
        df['wind_precipitation'] = df['precpt'] * (abs(df['avg_wind_level']) + 1)
    
    # Взаимодействие с праздниками
    if 'holiday_flag' in df.columns:
        df['holiday_temp'] = df['holiday_flag'] * df.get('avg_temperature', 0)
        df['holiday_activity'] = df['holiday_flag'] * df.get('activity_flag', 0)
    
    # Логарифмирование числовых признаков
    numeric_cols = ['precpt', 'avg_temperature', 'avg_humidity', 
                   'avg_wind_level', 'n_stores']
    
    for col in numeric_cols:
        if col in df.columns:
            # Добавляем небольшую константу, чтобы избежать log(0)
            df[f'log_{col}'] = np.log1p(np.abs(df[col])) * np.sign(df[col])
    
    # Квадраты и кубы важных признаков
    if 'avg_temperature' in df.columns:
        df['temp_squared'] = df['avg_temperature'] ** 2
        df['temp_cubed'] = df['avg_temperature'] ** 3
    
    if 'n_stores' in df.columns:
        df['stores_squared'] = df['n_stores'] ** 2
    
    # Статистики по группам (используем train_df для тестовых данных)
    if train_df is not None and 'price_p05' in train_df.columns:
        # Базовые статистики по основным группам
        group_cols_list = [
            ['first_category_id'], 
            ['second_category_id'], 
            ['third_category_id'], 
            ['product_id'],
            ['first_category_id', 'second_category_id']
        ]
        
        for group_col in group_cols_list:
            try:
                group_key = '_'.join(group_col) if len(group_col) > 1 else group_col[0]
                
                # Агрегация статистик
                agg_dict = {}
                if 'price_p05' in train_df.columns:
                    agg_dict['price_p05'] = ['mean', 'std', 'median']
                if 'price_p95' in train_df.columns:
                    agg_dict['price_p95'] = ['mean', 'std', 'median']
                if 'n_stores' in train_df.columns:
                    agg_dict['n_stores'] = ['mean', 'std']
                
                if agg_dict:
                    group_stats = train_df.groupby(group_col).agg(agg_dict)
                    
                    # Выравниваем многоуровневые колонки
                    group_stats.columns = [f'{col[0]}_{col[1]}_{group_key}' for col in group_stats.columns]
                    group_stats = group_stats.reset_index()
                    
                    # Объединяем с основным датафреймом
                    df = pd.merge(df, group_stats, on=group_col, how='left')
            except Exception as e:
                print(f"Warning: Failed to create stats for group {group_col}: {str(e)}")
                continue
    
    # Взаимодействия категориальных признаков (упрощенная версия)
    cat_cols = ['management_group_id', 'first_category_id', 
                'second_category_id', 'third_category_id']
    
    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            col1, col2 = cat_cols[i], cat_cols[j]
            if col1 in df.columns and col2 in df.columns:
                # Простое числовое взаимодействие
                df[f'{col1}_{col2}_sum'] = df[col1] + df[col2]
                df[f'{col1}_{col2}_prod'] = df[col1] * df[col2]
    
    # Заполнение пропусков
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            if train_df is not None and col in train_df.columns:
                fill_value = train_df[col].median()
            else:
                fill_value = df[col].median() if not df[col].empty else 0
            df[col].fillna(fill_value, inplace=True)
        elif df[col].dtype == 'object':
            df[col].fillna('unknown', inplace=True)
    
    # Удаляем возможные дубликаты колонок
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

# Создание расширенных признаков
print("Создание расширенных признаков для train...")
train_features = create_advanced_features(train)

print("Создание расширенных признаков для test...")
test_features = create_advanced_features(test, train_df=train)

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# Подготовка данных для моделирования
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import optuna

# Создаем кастомную метрику IoU для оптимизации
def iou_score(y_true_lower, y_true_upper, y_pred_lower, y_pred_upper, epsilon=1e-6):
    """Вычисляет IoU score для интервалов"""
    # Утолщение интервалов
    y_true_lower_adj = y_true_lower - epsilon
    y_true_upper_adj = y_true_upper + epsilon
    y_pred_lower_adj = y_pred_lower - epsilon
    y_pred_upper_adj = y_pred_upper + epsilon
    
    # Пересечение
    intersection_lower = np.maximum(y_true_lower_adj, y_pred_lower_adj)
    intersection_upper = np.minimum(y_true_upper_adj, y_pred_upper_adj)
    intersection = np.maximum(0, intersection_upper - intersection_lower)
    
    # Объединение
    union = (y_true_upper_adj - y_true_lower_adj) + \
            (y_pred_upper_adj - y_pred_lower_adj) - intersection
    
    return np.mean(intersection / np.maximum(union, epsilon))

# Подготовка данных для моделирования
print("\nПодготовка данных для моделирования...")

# Собираем числовые признаки
numeric_features = []
for col in train_features.columns:
    if train_features[col].dtype in ['float64', 'int64']:
        if col not in ['price_p05', 'price_p95', 'row_id', 'dt']:
            numeric_features.append(col)

print(f"Всего числовых признаков: {len(numeric_features)}")

X_train_full = train_features[numeric_features].fillna(0)
X_test_full = test_features[numeric_features].fillna(0)

y_train_p05 = train_features['price_p05']
y_train_p95 = train_features['price_p95']

# Отбор важных признаков с помощью RandomForest
print("Отбор важных признаков...")
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_full, (y_train_p05 + y_train_p95) / 2)

feature_importances = pd.Series(rf_selector.feature_importances_, index=numeric_features)
top_features = feature_importances.nlargest(80).index.tolist()  # Берем топ-80 признаков

print(f"Выбрано {len(top_features)} важных признаков")

X_train = X_train_full[top_features]
X_test = X_test_full[top_features]

# Масштабирование
print("Масштабирование признаков...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обнаружение и обработка аномалий
print("Обнаружение аномалий...")
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
anomalies = iso_forest.fit_predict(X_train_scaled)
normal_mask = anomalies == 1

X_train_normal = X_train_scaled[normal_mask]
y_train_p05_normal = y_train_p05.iloc[normal_mask].values
y_train_p95_normal = y_train_p95.iloc[normal_mask].values

print(f"Используем {len(X_train_normal)} нормальных наблюдений из {len(X_train_scaled)}")

# Кластеризация для создания дополнительных признаков
print("Кластеризация...")
n_clusters = min(7, len(X_train_scaled) // 100)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
train_clusters = kmeans.fit_predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# PCA для создания новых признаков
print("Применение PCA...")
pca = PCA(n_components=min(15, X_train_scaled.shape[1]))
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Собираем все признаки вместе
X_train_final = np.hstack([
    X_train_scaled[normal_mask],
    train_clusters[normal_mask].reshape(-1, 1),
    X_train_pca[normal_mask, :3],  # Берем первые 3 главных компоненты
    (X_train_scaled[normal_mask] ** 2)[:, :5]  # Квадраты первых 5 признаков
])

X_test_final = np.hstack([
    X_test_scaled,
    test_clusters.reshape(-1, 1),
    X_test_pca[:, :3],
    (X_test_scaled ** 2)[:, :5]
])

print(f"Финальная размерность признаков: {X_train_final.shape[1]}")

# Обучение ансамбля моделей
print("\n" + "="*50)
print("ОБУЧЕНИЕ АНСАМБЛЯ МОДЕЛЕЙ")
print("="*50)

# Функция для создания и обучения базовых моделей
def train_base_models(X_train, y_train, target_name):
    """Создает и обучает ансамбль базовых моделей"""
    print(f"\nОбучение моделей для {target_name}...")
    
    models = []
    
    # 1. XGBoost
    print("  Обучение XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models.append(('xgb', xgb_model))
    
    # 2. LightGBM
    print("  Обучение LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    models.append(('lgb', lgb_model))
    
    # 3. Random Forest
    print("  Обучение Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models.append(('rf', rf_model))
    
    # 4. Gradient Boosting
    print("  Обучение Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models.append(('gb', gb_model))
    
    # 5. Ridge Regression
    print("  Обучение Ridge Regression...")
    ridge_model = Ridge(alpha=1.0, random_state=42)
    ridge_model.fit(X_train, y_train)
    models.append(('ridge', ridge_model))
    
    return models

# Обучение моделей для price_p05
base_models_p05 = train_base_models(X_train_final, y_train_p05_normal, "price_p05")

# Обучение моделей для price_p95
base_models_p95 = train_base_models(X_train_final, y_train_p95_normal, "price_p95")

# Создание Stacking моделей
print("\nСоздание Stacking моделей...")

# Stacking для price_p05
print("  Stacking для price_p05...")
meta_model_p05 = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

stacking_model_p05 = StackingRegressor(
    estimators=base_models_p05,
    final_estimator=meta_model_p05,
    cv=3,
    n_jobs=-1
)
stacking_model_p05.fit(X_train_final, y_train_p05_normal)

# Stacking для price_p95
print("  Stacking для price_p95...")
meta_model_p95 = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

stacking_model_p95 = StackingRegressor(
    estimators=base_models_p95,
    final_estimator=meta_model_p95,
    cv=3,
    n_jobs=-1
)
stacking_model_p95.fit(X_train_final, y_train_p95_normal)

# Функция для ансамблевого предсказания
def ensemble_predict(X, base_models, stacking_model, weights=None):
    """Предсказание с помощью ансамбля моделей"""
    if weights is None:
        # Веса: базовые модели по 0.15 каждая, stacking - 0.25
        weights = [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
    
    predictions = np.zeros(len(X))
    
    # Предсказания базовых моделей
    for i, (name, model) in enumerate(base_models):
        predictions += weights[i] * model.predict(X)
    
    # Добавляем предсказание stacking модели
    predictions += weights[-1] * stacking_model.predict(X)
    
    return predictions

# Предсказание на тестовых данных
print("\nГенерация предсказаний...")

# Price_p05
test_predictions_p05 = ensemble_predict(
    X_test_final, 
    base_models_p05, 
    stacking_model_p05,
    weights=[0.12, 0.12, 0.12, 0.12, 0.12, 0.4]  # Stacking получает 40%
)

# Price_p95
test_predictions_p95 = ensemble_predict(
    X_test_final, 
    base_models_p95, 
    stacking_model_p95,
    weights=[0.12, 0.12, 0.12, 0.12, 0.12, 0.4]
)

# Расширенная постобработка
print("\nРасширенная постобработка...")

# Сохраняем копии оригинальных предсказаний
orig_p05 = test_predictions_p05.copy()
orig_p95 = test_predictions_p95.copy()

# 1. Корректируем интервалы
for i in range(len(test_predictions_p05)):
    # Проверяем, что p05 <= p95
    if test_predictions_p05[i] > test_predictions_p95[i]:
        # Если порядок нарушен, усредняем и создаем симметричный интервал
        midpoint = (test_predictions_p05[i] + test_predictions_p95[i]) / 2
        
        # Определяем ширину интервала на основе среднего значения
        if i > 0:
            # Используем среднюю ширину предыдущих интервалов
            recent_widths = []
            for j in range(1, min(6, i+1)):
                width = test_predictions_p95[i-j] - test_predictions_p05[i-j]
                if width > 0:
                    recent_widths.append(width)
            
            if recent_widths:
                avg_width = np.mean(recent_widths)
            else:
                avg_width = 0.05 * midpoint
        else:
            avg_width = 0.05 * midpoint  # 5% от средней цены
        
        # Гарантируем минимальную ширину
        avg_width = max(avg_width, 0.001)
        
        test_predictions_p05[i] = midpoint - avg_width / 2
        test_predictions_p95[i] = midpoint + avg_width / 2
    
    # 2. Минимальная и максимальная ширина интервала
    current_width = test_predictions_p95[i] - test_predictions_p05[i]
    midpoint = (test_predictions_p05[i] + test_predictions_p95[i]) / 2
    
    # Минимальная ширина (1% от средней цены, но не менее 0.001)
    min_width = max(0.01 * abs(midpoint), 0.001)
    
    # Максимальная ширина (20% от средней цены)
    max_width = 0.2 * abs(midpoint)
    
    if current_width < min_width:
        # Расширяем интервал
        extension = (min_width - current_width) / 2
        test_predictions_p05[i] -= extension
        test_predictions_p95[i] += extension
    elif current_width > max_width:
        # Сужаем интервал
        reduction = (current_width - max_width) / 2
        test_predictions_p05[i] += reduction
        test_predictions_p95[i] -= reduction

# 3. Сглаживание с помощью скользящего среднего
window_size = 3
if len(test_predictions_p05) > window_size:
    # Создаем копии для сглаживания
    smoothed_p05 = test_predictions_p05.copy()
    smoothed_p95 = test_predictions_p95.copy()
    
    for i in range(len(test_predictions_p05)):
        start = max(0, i - window_size // 2)
        end = min(len(test_predictions_p05), i + window_size // 2 + 1)
        
        if end - start > 1:
            smoothed_p05[i] = np.mean(test_predictions_p05[start:end])
            smoothed_p95[i] = np.mean(test_predictions_p95[start:end])
    
    # Применяем частичное сглаживание (50% оригинальных, 50% сглаженных)
    alpha = 0.5
    test_predictions_p05 = alpha * test_predictions_p05 + (1 - alpha) * smoothed_p05
    test_predictions_p95 = alpha * test_predictions_p95 + (1 - alpha) * smoothed_p95

# 4. Гарантируем, что цены неотрицательны
test_predictions_p05 = np.maximum(test_predictions_p05, 0.001)
test_predictions_p95 = np.maximum(test_predictions_p95, 0.001)

# Валидация на части train данных
print("\nВалидация модели...")

# Используем TimeSeriesSplit для валидации
tscv = TimeSeriesSplit(n_splits=3)
cv_scores_iou = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_final)):
    # Разделение данных
    X_fold_train = X_train_final[train_idx]
    X_fold_val = X_train_final[val_idx]
    
    y_fold_p05_train = y_train_p05_normal[train_idx]
    y_fold_p95_train = y_train_p95_normal[train_idx]
    
    y_fold_p05_val = y_train_p05_normal[val_idx]
    y_fold_p95_val = y_train_p95_normal[val_idx]
    
    # Обучение XGBoost на фолде (упрощенная модель для валидации)
    xgb_fold_p05 = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_fold_p95 = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    xgb_fold_p05.fit(X_fold_train, y_fold_p05_train)
    xgb_fold_p95.fit(X_fold_train, y_fold_p95_train)
    
    # Предсказания
    pred_p05 = xgb_fold_p05.predict(X_fold_val)
    pred_p95 = xgb_fold_p95.predict(X_fold_val)
    
    # IoU метрика
    iou = iou_score(y_fold_p05_val, y_fold_p95_val, pred_p05, pred_p95)
    cv_scores_iou.append(iou)
    
    print(f"Fold {fold+1}: IoU = {iou:.4f}")

print(f"\nСредний IoU по кросс-валидации: {np.mean(cv_scores_iou):.4f} (±{np.std(cv_scores_iou):.4f})")

# Создание submission файла
print("\nСоздание submission файла...")

submission = pd.DataFrame({
    'row_id': test_features['row_id'],
    'price_p05': test_predictions_p05,
    'price_p95': test_predictions_p95
})

# Сохранение submission файла
submission_path = 'E:\\submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n✓ Submission файл сохранен по пути: {submission_path}")
print(f"✓ Размер submission: {submission.shape}")
print(f"✓ Колонки в submission: {submission.columns.tolist()}")

# Детальная диагностика
print("\n" + "="*50)
print("ДИАГНОСТИКА ПРЕДСКАЗАНИЙ")
print("="*50)

print(f"\nСтатистика price_p05:")
print(f"  Min: {submission['price_p05'].min():.6f}")
print(f"  Mean: {submission['price_p05'].mean():.6f}")
print(f"  Max: {submission['price_p05'].max():.6f}")
print(f"  Std: {submission['price_p05'].std():.6f}")

print(f"\nСтатистика price_p95:")
print(f"  Min: {submission['price_p95'].min():.6f}")
print(f"  Mean: {submission['price_p95'].mean():.6f}")
print(f"  Max: {submission['price_p95'].max():.6f}")
print(f"  Std: {submission['price_p95'].std():.6f}")

# Ширина интервалов
widths = submission['price_p95'] - submission['price_p05']
print(f"\nСтатистика ширины интервалов:")
print(f"  Min width: {widths.min():.6f}")
print(f"  Mean width: {widths.mean():.6f}")
print(f"  Max width: {widths.max():.6f}")
print(f"  Median width: {widths.median():.6f}")
print(f"  Zero/negative widths: {(widths <= 0).sum()}")

# Проверка корректности
invalid_intervals = (submission['price_p05'] > submission['price_p95']).sum()
print(f"\n✓ Некорректных интервалов (p05 > p95): {invalid_intervals}")

if invalid_intervals > 0:
    print("Исправление некорректных интервалов...")
    mask = submission['price_p05'] > submission['price_p95']
    
    for idx in submission[mask].index:
        avg = (submission.loc[idx, 'price_p05'] + submission.loc[idx, 'price_p95']) / 2
        width = max(0.01 * avg, 0.001)  # 1% ширины, минимум 0.001
        submission.loc[idx, 'price_p05'] = avg - width / 2
        submission.loc[idx, 'price_p95'] = avg + width / 2
    
    submission.to_csv(submission_path, index=False)
    print("✓ Submission обновлен")

# Пример первых 10 предсказаний
print(f"\nПримеры предсказаний (первые 10 строк):")
print(submission.head(10).to_string())

print("\n" + "="*50)
print("✓ ВСЕ ОПЕРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО!")
print("="*50)
