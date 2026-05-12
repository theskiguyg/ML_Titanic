from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import yaml 
import pandas as pd 
from sklearn.pipeline import Pipeline
from preprocess import get_preprocessor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import os
import joblib

def _save_model(pipeline, config):
    save_path = config['model'].get('save_path', 'models/model.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"✅ Model saved to {save_path}")

def start_train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data = pd.read_csv(config['data']['train_path'])

    if 'drop_cols' in config['preprocess']:
        cols_to_drop = [col for col in config['preprocess']['drop_cols'] if col in data.columns]
        data = data.drop(columns=cols_to_drop)


    X_train = data.drop(config['data']['target'], axis=1)
    y_train = data[config['data']['target']]

    preprocessor = get_preprocessor(
        config['preprocess']['num_cols'], 
        config['preprocess']['cat_cols']
        )
    
    model_name = config['model']['name']
    if model_name == 'SGD':
        model = SGDClassifier(**config['SGD'])
    elif model_name == 'RF':
        model = RandomForestClassifier(**config['RF'])
        
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor), 
        ('model', model)
    ])

    if config['cv']['do']:
        scores = cross_val_score(
            full_pipeline, X_train, y_train, 
            cv=config['cv']['folds'], 
            scoring=config['cv'].get('scoring', 'accuracy')
            )
        print('CrossValidation is done')
        print(f'CV scores: {scores}')
        full_pipeline.fit(X_train, y_train)

    elif config['search']['do']:
        param_grid = config['search']['param_grid']
        grid_search = GridSearchCV(
            full_pipeline,
            param_grid=param_grid, 
            cv=config['search']['cv'], 
            scoring=config['search']['scoring']
            )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print('GridSearchCV is done')
        print(f"Best params: {best_params}")
        print(f"Best CV scoring: {best_score:.4f}")
        full_pipeline.fit(X_train, y_train)

    else:
        full_pipeline.fit(X_train, y_train)
        print('Model trained')
        predictions = full_pipeline.predict(X_train)
        scoring = accuracy_score(y_train, predictions)
        print(f"Accuracy scoring: {scoring}")

        if config['model'].get('save', True):
            _save_model(full_pipeline, config)
        else:
            print('Model is not saved!')

if __name__ == "__main__":
    start_train('configs/baseline.yaml')

