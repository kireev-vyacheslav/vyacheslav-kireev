import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
train = pd.read_csv('train_kaggle.csv')
test = pd.read_csv('test_kaggle.csv')

print("Размер train:", train.shape)
print("Размер test:", test.shape)

# Анализ эффективности действий
action_stats = train.groupby('segment').agg({
    'visit': ['mean', 'count']
})
print("Эффективность действий:")
print(action_stats)

# Определение лучшей статической политики
best_static_action = action_stats[('visit', 'mean')].idxmax()
best_static_value = action_stats[('visit', 'mean')].max()
print(f"Лучшая статическая политика: {best_static_action} = {best_static_value:.4f}")

# Предобработка данных
def prepare_features(df):
    df_processed = df.copy()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    for col in ['segment', 'id']:
        if col in categorical_cols:
            categorical_cols.remove(col)
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed

train_processed = prepare_features(train)
test_processed = prepare_features(test)

feature_cols = [col for col in train_processed.columns 
                if col not in ['id', 'segment', 'visit']]

print(f"Количество признаков: {len(feature_cols)}")

# Основная функция оптимизации политики
def hyper_aggressive_policy(train_df, test_df, feature_cols):
    actions = ['Mens E-Mail', 'Womens E-Mail', 'No E-Mail']
    
    # Обучение моделей для каждого действия
    action_models = {}
    action_performance = {}
    
    for action in actions:
        action_data = train_df[train_df['segment'] == action]
        action_performance[action] = action_data['visit'].mean()
        
        if len(action_data) > 30:
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(action_data[feature_cols], action_data['visit'])
            action_models[action] = model
    
    # Предсказание для тестовых данных
    test_predictions = np.zeros((len(test_df), len(actions)))
    
    for i, action in enumerate(actions):
        if action in action_models:
            test_predictions[:, i] = action_models[action].predict_proba(test_df[feature_cols])[:, 1]
        else:
            test_predictions[:, i] = action_performance[action]
    
    # Оптимизация распределения вероятностей
    final_policy = np.zeros((len(test_df), len(actions)))
    
    for i in range(len(test_df)):
        scores = test_predictions[i]
        
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        other_scores = [scores[j] for j in range(len(scores)) if j != best_idx]
        min_diff = best_score - max(other_scores) if other_scores else 0.1
        
        # Адаптивное распределение вероятностей на основе уверенности
        if min_diff > 0.03:
            final_policy[i, best_idx] = 0.995
            remaining = 0.005 / (len(actions) - 1)
            for j in range(len(actions)):
                if j != best_idx:
                    final_policy[i, j] = remaining
                    
        elif min_diff > 0.015:
            final_policy[i, best_idx] = 0.98
            remaining = 0.02 / (len(actions) - 1)
            for j in range(len(actions)):
                if j != best_idx:
                    final_policy[i, j] = remaining
                    
        elif min_diff > 0.008:
            final_policy[i, best_idx] = 0.95
            remaining = 0.05 / (len(actions) - 1)
            for j in range(len(actions)):
                if j != best_idx:
                    final_policy[i, j] = remaining
                    
        elif min_diff > 0.004:
            final_policy[i, best_idx] = 0.90
            remaining = 0.10 / (len(actions) - 1)
            for j in range(len(actions)):
                if j != best_idx:
                    final_policy[i, j] = remaining
                    
        else:
            temperature = 0.05
            exp_scores = np.exp((scores - np.max(scores)) / temperature)
            probs = exp_scores / np.sum(exp_scores)
            
            probs[best_idx] = max(0.80, probs[best_idx])
            probs = probs / np.sum(probs)
            
            final_policy[i] = probs
    
    return final_policy, actions

# Базовые вероятности для стабильности
def create_baseline_policy(n_samples):
    actions = ['Mens E-Mail', 'Womens E-Mail', 'No E-Mail']
    baseline = np.zeros((n_samples, len(actions)))
    
    for i in range(n_samples):
        baseline[i, 0] = 0.85
        baseline[i, 1] = 0.10
        baseline[i, 2] = 0.05
    
    return baseline

# Запуск основной функции
final_policy, actions = hyper_aggressive_policy(train_processed, test_processed, feature_cols)

baseline_policy = create_baseline_policy(len(test_processed))

# Комбинирование агрессивной и базовой политик
combined_policy = 0.9 * final_policy + 0.1 * baseline_policy

# Создание финального submission
action_map = {
    'Mens E-Mail': 'p_mens_email',
    'Womens E-Mail': 'p_womens_email', 
    'No E-Mail': 'p_no_email'
}

submission = pd.DataFrame({'id': test['id']})
for i, action in enumerate(actions):
    col_name = action_map[action]
    submission[col_name] = combined_policy[:, i]

# Нормализация вероятностей
submission[['p_mens_email', 'p_womens_email', 'p_no_email']] = (
    submission[['p_mens_email', 'p_womens_email', 'p_no_email']]
    .div(submission[['p_mens_email', 'p_womens_email', 'p_no_email']].sum(axis=1), axis=0)
)

# Анализ результатов
print("Статистика распределения вероятностей:")
aggression_levels = {
    'ULTRA (>99%)': (submission[['p_mens_email', 'p_womens_email', 'p_no_email']] > 0.99).any(axis=1).sum(),
    'SUPER (>95%)': (submission[['p_mens_email', 'p_womens_email', 'p_no_email']] > 0.95).any(axis=1).sum(),
    'HIGH (>90%)': (submission[['p_mens_email', 'p_womens_email', 'p_no_email']] > 0.9).any(axis=1).sum(),
    'MEDIUM (>80%)': (submission[['p_mens_email', 'p_womens_email', 'p_no_email']] > 0.8).any(axis=1).sum(),
    'BALANCED': ((submission[['p_mens_email', 'p_womens_email', 'p_no_email']] <= 0.8).all(axis=1)).sum()
}

for level, count in aggression_levels.items():
    percentage = count / len(submission) * 100
    print(f"{level}: {count} пользователей ({percentage:.1f}%)")

print("Средние вероятности по действиям:")
for col in ['p_mens_email', 'p_womens_email', 'p_no_email']:
    mean_prob = submission[col].mean()
    print(f"{col}: {mean_prob:.4f}")

# Проверка корректности вероятностей
sums = submission[['p_mens_email', 'p_womens_email', 'p_no_email']].sum(axis=1)
print(f"Проверка сумм вероятностей: min={sums.min():.10f}, max={sums.max():.10f}")

# Сохранение результатов
submission.to_csv('E:\\submission.csv', index=False)
print("Файл submission.csv сохранен")
