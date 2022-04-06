from .base_model import BaseModel


class WideDeep(BaseModel):
    def __init__(self, **params):
        super(WideDeep, self).__init__(**params)

    def _forward(self, features, is_training):
        pass
