from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
def create_pipeline(preprocessor):
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('rf', RandomForestClassifier())
        ]
    )
    return pipeline