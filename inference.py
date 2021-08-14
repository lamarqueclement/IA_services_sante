import json

from commons import get_model, transform_image
from PIL import Image
import torch
import onnx

imagenet_class_index = json.load(open('imagenet_class_index.json'))

def sigmoid(x):
    from numpy import exp
    return 1/(1+exp(-x))

def to_figure(results_mod):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig = go.Figure(data=[
        go.Bar(name='Results', x=list(results_mod.keys()), y=list(results_mod.values()))
        ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    
    fig.update_layout(
            title=f"Scores obtenus par les différentes catégories pour l'image",
            xaxis_title="Catégories",
            yaxis_title="Score",
            legend_title=" ",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="RebeccaPurple"
            )
        )  
    fig.show()
    fig.write_html(f"static/graph_results.html")
    fig.write_image(f"static/image_results.png")
    

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_prediction(image_bytes, ort_session):
    try:
        img_y = transform_image(image_bytes)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[0]
    except Exception:
        return 404, 'error'
    return [imagenet_class_index.get(str(outputs.argmax())),outputs.argmax(),outputs[0]]

