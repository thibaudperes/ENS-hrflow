from hrflow_recsys.api import create_app, load_demo, load_model

app = create_app(load_model(), load_demo())
