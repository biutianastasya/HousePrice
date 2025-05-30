import os
from flask import Flask

def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'templates'),
        static_folder=os.path.join(base_dir, 'static')
    )
    app.config['MODEL_PATH']    = 'app/property_price_model.pkl'
    app.config['FEATURES_PATH'] = 'app/property_price_features.pkl'
    app.config['POI_PATTERN']   = 'app/poi_*.csv'

    from app.controllers import main
    app.register_blueprint(main)
    return app
