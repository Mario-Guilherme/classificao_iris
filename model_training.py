import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

from data_source import DataSource
from data_treatment import DataTreatment
from preprocessing import Preprocessing

class ModelTtrainning:

    def __init__(self):
        self._data = DataSource()
        self._data_treatment = DataTreatment()
        self._preprocessing = Preprocessing()
        self._model = DecisionTreeClassifier(criterion='entropy', random_state=0)

    def model_trainning(self):
        
        print("Carregando os dados e tratando os dados")
        df = self._data_treatment.data_treatment(self._data.read_data())
        
        print("Preprocessando os dados")
        X_train, y_train = self._preprocessing.pre_processing(df)

        print("Treinando o modelo")
        self._model.fit(X_train, y_train)

        model = {'model_obj' : self._model,
                 'preprocessing' : self._preprocessing,
                 'colunas' : self._preprocessing._feature_names}
        print(model)
        dump(model, 'modelo.pkl')

        return model    
