import pandas as pd
import joblib


class Utils:

    @staticmethod
    def load_from_csv(path):
        return pd.read_csv(path)

    def load_from_sql(self):
        pass

    @staticmethod
    def features_target(dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y

    @staticmethod
    def model_export(clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')
