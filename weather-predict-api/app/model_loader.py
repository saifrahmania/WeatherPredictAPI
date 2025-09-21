import os, joblib

class ModelStore:
    _clf = None
    _reg = None

    @classmethod
    def load(cls):
        if cls._clf is None:
            cls._clf = joblib.load(os.path.join("models", "rain_or_not.pkl"))
        if cls._reg is None:
            cls._reg = joblib.load(os.path.join("models", "precipitation_fall.pkl"))
        return cls._clf, cls._reg
