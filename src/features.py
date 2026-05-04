import pandas as pd
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         X = X.copy()
#         X['FamilySize'] = X['Parch'] + X['SibSp'] + 1
#         X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
#         return X


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Добавляет фичи FamilySize и IsAlone.
    Работает с numpy array. Предполагает, что колонки Parch и SibSp
    находятся на определенных позициях (по умолчанию последние две).
    """
    def __init__(self, parch_idx=-2, sibsp_idx=-1):
        self.parch_idx = parch_idx
        self.sibsp_idx = sibsp_idx
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X — numpy array
        # Берем колонки по индексам
        parch = X[:, self.parch_idx]
        sibsp = X[:, self.sibsp_idx]
        
        # Создаем новые признаки
        family_size = parch + sibsp + 1
        is_alone = (family_size == 1).astype(int)
        
        # Добавляем как новые колонки (в конец)
        X_new = np.column_stack([X, family_size, is_alone])
        
        return X_new