import base64
import io
from PIL import Image
import easyocr
from pdf2image import convert_from_bytes
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from openai import OpenAI
import os
import numpy as np
import dash_bootstrap_components as dbc
from flask import Flask, Response
from flask_basicauth import BasicAuth


client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

server = Flask(__name__)

server.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME') # Replace with your desired username
server.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')  # Replace with your desired password
server.config['BASIC_AUTH_FORCE'] = True  # This will force BasicAuth on all routes

basic_auth = BasicAuth(server)

app = dash.Dash(__name__,  server=server, external_stylesheets=[dbc.themes.BOOTSTRAP],   meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'},
        # MathJax CDN for LaTeX rendering
        {'src': 'https://polyfill.io/v3/polyfill.min.js?features=es6', 'type': 'text/javascript'},
        {'src': 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML', 'type': 'text/javascript'}
    ] )

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
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='image-upload-confirm', style={'fontSize': '16px', 'color': 'green', 'margin': '10px 0'}), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Button('RUN', id='translate-button', style={'fontSize': '24px', 'padding': '10px 20px', 'margin': '20px 0'}), width=6),
        dbc.Col(html.Button('Clear Image', id='clear-button', style={'fontSize': '24px', 'padding': '10px 20px', 'margin': '20px 0'}), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(
            id="loading-output",
            children=html.Div(id='output-image-upload', style={'whiteSpace': 'pre-line',  'overflowY': 'auto'}),
            type='circle',
            color="#00CC96"        ), width=12)
    ]),
], fluid=True)

def process_with_openai(text_pic, input_system, input_user):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": input_system},
            {"role": "user", "content": input_user},
            {"role": "assistant", "content": text_pic}
        ]
    )
    return response.choices[0].message.content

def easyocr_reader(image):
    reader = easyocr.Reader(["ru","rs_cyrillic","uk","en"], gpu=False)  # Assume no GPU available
    results = reader.readtext(np.array(image))
    return ' '.join([result[1] for result in results])

@app.callback(
    Output('upload-image', 'contents'),
    [Input('clear-button', 'n_clicks')]
)
def clear_uploaded_file(n_clicks):
    if n_clicks:
        return None
    return dash.no_update

# Only update if the button is clicked

@app.callback(
    Output('image-upload-confirm', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_image_upload(contents, filename):
    if contents:
        # Generate the image preview
        return html.Div([
            html.P(f"Successfully uploaded {filename}!"),
            # Display the image as a thumbnail
            html.Img(src=contents, style={'height': '100px', 'margin': '10px'})  # Adjust size as needed
        ])

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('translate-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('upload-image', 'filename'),
     State('input_system', 'value'),
     State('input_user', 'value')]
)
def update_output(n_clicks, contents, filename, input_system, input_user):
    if n_clicks is None:
        return dash.no_update
    try:
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

        processed_text = process_with_openai(text_pic, input_system, input_user)
        return  html.Div([
                    html.H5("Extracted Text from Image:"),
                    dcc.Markdown('```' + (text_pic or "No image uploaded or text extracted.") + '```', dangerously_allow_html=True),
                    html.H5("Processed Response:"),
                    dcc.Markdown(processed_text, dangerously_allow_html=True)
                ], style={'width': '100%', 'margin': '20px 0', 'whiteSpace': 'pre-line'})


    except Exception as e:
        return html.Div(['Error processing request: {}'.format(e)])



if __name__ == '__main__':
    app.run_server(debug=True)
