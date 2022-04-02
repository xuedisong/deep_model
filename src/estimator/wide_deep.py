from .base_model import BaseModel


class WideDeep(BaseModel):
    def __init__(self, **params):
        super(WideDeep, self).__init__(**params)

    def _forward(self, feature):
        pass

    def get_model(self, features, labels, mode, params):
        pass
