import yaml 
import pandas as pd 
import joblib 
import os 
from sklearn.metrics import accuracy_score

def load_config(config_path: str):
    with open(config_path, 'r') as f: 
        return yaml.safe_load(f)
    
def load_model(model_path: str): 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found. Save the model to {model_path}")
    return joblib.load(model_path)


def start_predict(config_path: str):
    config = load_config(config_path)
    model_path =  config['predict'].get('model_path')

    data = pd.read_csv(config['data']['test_path'])
    gender_sub = pd.read_csv(config['data']['test_target_path'])

    X_test = data.drop('PassengerId', axis=1)
    y_test = gender_sub['Survived']


    model = load_model(model_path)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {acc:.4f}")

    if config['predict']['save_predictions']:
        submit_path = config['predict'].get('submit_path', 'data/submissions/submission.csv')
        os.makedirs(os.path.dirname(submit_path), exist_ok=True)
        submission = pd.DataFrame({
            'prediction': predictions
        })
        submission.to_csv(submit_path, index=False)
        print(f"Submission saved to {submit_path}")

if __name__=='__main__':
    start_predict('configs/baseline.yaml')





