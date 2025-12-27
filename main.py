"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""



def create_submission(submission_df):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # 1. Загрузка данных (без указания типов)
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    print(f"Train размер: {train_df.shape}")
    print(f"Test размер: {test_df.shape}")

    # 2. Создание минимальных признаков

    def create_simple_features(df):
        """Создание простейших признаков"""
        features = pd.DataFrame(index=df.index)
    
    # 1. Длина запроса и заголовка
        features['query_len'] = df['query'].str.len()
        features['title_len'] = df['product_title'].str.len()
    
    # 2. Количество слов
        features['query_words'] = df['query'].str.split().str.len()
        features['title_words'] = df['product_title'].str.split().str.len()
    
    # 3. Простое совпадение слов
        features['word_match'] = df.apply(
            lambda x: len(set(str(x['query']).lower().split()) & 
                        set(str(x['product_title']).lower().split())), 
            axis=1
        )   
    
    # 4. Нормализованное совпадение
        features['word_match_ratio'] = features['word_match'] / (features['query_words'] + 1)
    
    # 5. Бренд в запросе (бинарный)
        features['brand_in_query'] = df.apply(
            lambda x: 1 if pd.notna(x.get('product_brand')) and 
                     str(x['product_brand']).lower() in str(x['query']).lower() else 0,
            axis=1
        )
    
    # 6. Наличие данных
        features['has_brand'] = df['product_brand'].notna().astype(int)
        features['has_color'] = df['product_color'].notna().astype(int)
    
        return features

# Создаем признаки
    X_train = create_simple_features(train_df)
    X_test = create_simple_features(test_df)

    print(f"Признаки созданы: {X_train.shape[1]} признаков")

# 3. Подготовка для LightGBM

# Целевая переменная
    y_train = train_df['relevance']

# Кодируем query_id если они строковые
    if train_df['query_id'].dtype == 'object':
        le = LabelEncoder()
        all_ids = pd.concat([train_df['query_id'], test_df['query_id']])
        le.fit(all_ids)
        train_groups = le.transform(train_df['query_id'])
    else:
        train_groups = train_df['query_id'].values

# Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# 4. Обучение модели

    params = {
        'objective': 'rank:ndcg',  # Ранжирование с NDCG метрикой
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 993,
        'verbosity': 0,
        'eval_metric': 'ndcg@10',
        'tree_method': 'hist'  # Быстрый метод
    }

    # Создаем Dataset
    train_data = xgb.DMatrix(
        X_train_scaled,
        label=y_train,
        qid=train_groups
    )

    # Обучение
    model = xgb.train(
        params,
        train_data,
        num_boost_round=300,
        verbose_eval=50
    )

# 5. Предсказание
    test_dmatrix = xgb.DMatrix(X_test_scaled)
    predictions = model.predict(test_dmatrix)
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': predictions
    })
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission_df)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()
