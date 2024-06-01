import base64
import io
import threading
from PIL import Image

from pdf2image import convert_from_bytes
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
from openai import OpenAI
import os
import numpy as np
import dash_bootstrap_components as dbc
from flask import Flask
from flask_basicauth import BasicAuth
from google.cloud import vision

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


server = Flask(__name__)

server.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')  # Replace with your desired username
server.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')  # Replace with your desired password
server.config['BASIC_AUTH_FORCE'] = True
# This will force BasicAuth on all routes

basic_auth = BasicAuth(server)

app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP], meta_tags=[
    {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
    {'src': 'https://polyfill.io/v3/polyfill.min.js?features=es6', 'type': 'text/javascript'},
    {'src': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML',
     'type': 'text/javascript'}
])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Textarea(
            id='input_system',
            placeholder="Input your system instructions here",
            style={'width': '100%', 'height': '150px', 'fontSize': 16}
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Textarea(
            id='input_user',
            placeholder="Input your user instructions here",
            style={'width': '100%', 'height': '200px', 'fontSize': 16}
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('select files')]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                   'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='image-upload-confirm', style={'fontSize': '16px', 'color': 'green', 'margin': '10px 0'}),
                width=12)
    ]),
    dbc.Row([
        dbc.Col(dbc.Input(
            id='image_size',
            type='text',
            placeholder="Enter image size (1024x1024, 1792x1024, 1024x1792)",
            style={'width': '100%', 'fontSize': 16}
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col(dbc.Input(
            id='image_quality',
            type='text',
            placeholder="Enter image quality (standard or hd)",
            style={'width': '100%', 'fontSize': 16}
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Button('RUN', id='translate-button',
                            style={'fontSize': '24px', 'padding': '10px 20px', 'margin': '20px 0'}), width=4),
        dbc.Col(html.Button('Stop', id='stop-button',
                            style={'fontSize': '24px', 'padding': '10px 20px', 'margin': '20px 0'}), width=4),
        dbc.Col(html.Button('Clear Image', id='clear-button',
                            style={'fontSize': '24px', 'padding': '10px 20px', 'margin': '20px 0'}), width=4),
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(
            id="loading-output",
            children=html.Div(id='output-image-upload', style={'whiteSpace': 'pre-line', 'overflowY': 'auto'}),
            type='circle',
            color="#00CC96"), width=12)
    ]),
], fluid=True)

stop_flag = threading.Event()


def process_with_openai(text_pic, input_system, input_user):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": input_system},
            {"role": "user", "content": input_user},
            {"role": "assistant", "content": text_pic}
        ]
    )
    if stop_flag.is_set():
        return "Request terminated by user."
    return response.choices[0].message.content


def generate_image(prompt, size="1024x1024", quality="standard"):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality
    )

    if stop_flag.is_set():
        return "Request terminated by user."
    return response.data[0].url


def easyocr_reader(image):
    # Initialize the client
    client = vision.ImageAnnotatorClient()

    # Convert the image to bytes
    image = Image.fromarray(np.array(image))
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    content = buffered.getvalue()

    # Create an image object
    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Check for errors in the response
    if response.error.message:
        raise Exception(f'{response.error.message}')

    # Extract the detected text
    results = [text.description for text in texts]

    return ' '.join(results)


@app.callback(
    Output('upload-image', 'contents'),
    [Input('clear-button', 'n_clicks')]
)
def clear_uploaded_file(n_clicks):
    if n_clicks:
        return None
    return dash.no_update


@app.callback(
    Output('image-upload-confirm', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_image_upload(contents, filename):
    if contents:
        return html.Div([
            html.P(f"Successfully uploaded {filename}!"),
            html.Img(src=contents, style={'height': '100px', 'margin': '10px'})  # Adjust size as needed
        ])


@app.callback(
    Output('output-image-upload', 'children'),
    [Input('translate-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('upload-image', 'filename'),
     State('input_system', 'value'),
     State('input_user', 'value'),
     State('image_size', 'value'),
     State('image_quality', 'value')]
)
def update_output(n_clicks, contents, filename, input_system, input_user, image_size, image_quality):
    if n_clicks is None:
        return dash.no_update
    try:
        stop_flag.clear()
        text_pic = ""
        if contents:
            content_type, content_string = contents.split(',', 1)  # Correctly splits the base64 prefix from the data
            decoded = base64.b64decode(content_string)
            if 'pdf' in filename:
                images = convert_from_bytes(decoded)
                text_pic = "".join([easyocr_reader(image) for image in images])
            else:
                image = Image.open(io.BytesIO(decoded))
                text_pic = easyocr_reader(image)

        valid_sizes = ["1024x1024", "1792x1024", "1024x1792"]
        if "picture" in input_system.lower():
            if image_size not in valid_sizes:
                image_size = "1024x1024"
            if image_quality not in ["standard", "hd"]:
                image_quality = "standard"
            image_url = generate_image(input_user, image_size, image_quality)
            return html.Div([
                html.H5("Generated Image:"),
                html.Img(src=image_url, style={'width': '80%', 'hight':'80%', 'margin': '30px 0'})
            ])

        processed_text = process_with_openai(text_pic, input_system, input_user)
        return html.Div([
            html.H5("Extracted Text from Image:"),
            dcc.Markdown('```' + (text_pic or "No image uploaded or text extracted.") + '```',
                         dangerously_allow_html=True),
            html.H5("Processed Response:"),
            dcc.Markdown(processed_text, dangerously_allow_html=True)
        ], style={'width': '100%', 'margin': '20px 0', 'whiteSpace': 'pre-line'})

    except Exception as e:
        return html.Div(['Error processing request: {}'.format(e)])


@app.callback(
    Output('output-image-upload', 'children', allow_duplicate=True),
    [Input('stop-button', 'n_clicks')],
    prevent_initial_call=True
)
def stop_request(n_clicks):
    if n_clicks:
        stop_flag.set()
        return html.Div(['Request terminated by user.'])


if __name__ == '__main__':
    app.run_server(debug=True)
