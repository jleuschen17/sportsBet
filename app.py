import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from jupyter_dash import JupyterDash
import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import numpy as np
# import os
import openai
import gspread
import shutil
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('python-sheets-2-357617-873449e7f558 (1).json', scope)
client = gspread.authorize(creds)
# auth.authenticate_user()
# from google.auth import default
# creds, _ = default()
# gc = gspread.authorize(creds)

pwords = {'jshiesty' : 'Q9xt2ipwtv'}
inputTxt = ""
outputTxt = ""
models = {'parlay' : 'text-davinci-002',
          'straight' : 'code-davinci-002',
          'prop' : 'text-XXXX-002',
          'hedge' : 'text-XXXX-002',
          'Cover' : 'text-XXXX-002'}
inputs = []
outputs = []
sheetOutputs = []
def simple_request(prompt,
                  model="text-davinci-002",
                  temperature=0.5,
                  max_tokens=2500,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0):
  response = openai.Completion.create(
    model=model,
    prompt=prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty
  )
  return response["choices"][0]["text"]

def get_sheet(sheetName, index):
  sheet = client.open(sheetName)
  sheet_instance = sheet.get_worksheet(index)
  return sheet_instance

def checkStatus(user, pw, locks, clicks, style, price, margin, sport, clicks2):
    args = [user, pw, locks, clicks, style, price, margin, sport, clicks2]
    strArgs = ["user", "pw", "locks", "clicks", "style", "price", "margin", "sport", "clicks2"]
    statusStr = ""
    for i in range(len(args)):
        statusStr += f"{strArgs[i]}: {str(args[i])}\n"
    statusStr += f"length inputs: {len(inputs)}\n"
    statusStr += f"length outputs: {len(outputs)}\n"
    statusStr += f"length sheetOutputs: {len(sheetOutputs)}\n"
    statusStr += f"inputTxt: {inputTxt}\n"
    statusStr += f"outputTxt: {outputs}\n"
    return statusStr
def get_df(sheetName, index):
  sheet_instance = get_sheet(sheetName, index)
  data = sheet_instance.get_all_records()
  df = pd.DataFrame.from_dict(data)
  return df


def convert_to_txt(string, filename, folder=False):
  with open(filename, 'w', errors='ignore') as f:
    f.write(string)

def file_to_string(filename):
    file = open(filename, 'r', errors='ignore')
    string = file.read()
    return string

def generate_reponse_file(input, output, model, temp):
    prompt = file_to_string("RodWave.txt")
    response = simple_request(prompt, model="code-davinci-002", temperature=0.99, max_tokens=4000)
    convert_to_txt(response, "/content/drive/MyDrive/Colab Notebooks/NLP Engine/nbashit.txt")

def save_to_drive_output(file, folder=False):
  if not folder:
    shutil.move(file, f"/content/drive/MyDrive/Colab Notebooks/NLP Engine/Output Data/{file}")
  else:
    shutil.move(file, f"/content/drive/MyDrive/Colab Notebooks/NLP Engine/Output Data/{folder}/{file}")

def save_to_drive_input(file):
  shutil.move(file, f"/content/drive/MyDrive/Colab Notebooks/NLP Engine/Input Data/{file}")

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1('jshiestys sportsbook', style={'text-align' : 'center'}),

    html.H5('key', style={'text-align' : 'left'}),

    dcc.Input(id='key', type='text', debounce=True,
              multiple=False, style={'marginRight' : '10px', },
              placeholder=''),

    html.Br(),

    html.H5('user', style={'text-align' : 'left'}),

    dcc.Input(id='user', type='text', debounce=True,
              multiple=False, style={'marginRight' : '10px', },
              placeholder=''),

    html.H5('pw', style={'text-align' : 'left'}),

    dcc.Input(id='pw', type='text', debounce=True,
              multiple=False, style={'marginRight' : '10px', },
              placeholder=''),

    html.Br(),

    html.H5('sport', style={'text-align' : 'left'}),

    dcc.Input(id='sport', type='text', debounce=True,
              multiple=False, style={'marginRight' : '10px', },
              placeholder=''),

    html.Br(),

    html.H5('select bet type', style={'text-align' : 'left'}),

    dcc.Dropdown(id="style",
                 options=[{'label':i, 'value': models[i]} for i in models],
                 multi=False),

    html.Br(),

    html.H5('set value', style={'text-align' : 'left'}),

    dcc.Input(id='price', type='number', debounce=True,
              multiple=False, style={'marginRight' : '10px', },
              placeholder=''),

    html.Br(),

    html.H5('margin value', style={'text-align' : 'left'}),

    dcc.Input(id='margin', type='number', debounce=True,
              multiple=False, style={'marginRight' : '10px'},
              value=0),

    html.Br(),

    html.H5('spread', style={'text-align' : 'left'}),

    dcc.Input(id='spread', type='number', debounce=True,
              multiple=False, style={'marginRight' : '10px'},
              value=1.00),

    html.Br(),

    html.H5('line', style={'text-align' : 'left'}),

    dcc.Input(id='line', type='number', debounce=True,
              multiple=False, style={'marginRight' : '10px'},
              value = 0.00),

    html.Br(),

    html.H5('O/U', style={'text-align' : 'left'}),

    dcc.Input(id='overUnder', type='number', debounce=True,
              multiple=False, style={'marginRight' : '10px'},
              value = 0.00),

    html.H5('bets', style={'text-align' : 'left'}),

    dcc.Input(id='locks', type='text', debounce=True,
              multiple=False, style={'marginRight' : '10px', },
              placeholder=''),

    html.Br(),

    html.H5('      ', style={'text-align' : 'left'}),

    html.Button('add bet', id='createBetButton', n_clicks=0),

    html.Br(),

    html.Div(id='componentOutput', style={'whiteSpace': 'pre-line'}),

    html.Br(),

    html.Div(id='componentOutput2', style={'whiteSpace' : 'pre-line'}),

    html.Br(),

    html.Button('confirm bet', id='confirmButton', n_clicks=0),
])

@app.callback(
    Output('componentOutput', 'children'),
    Output('componentOutput2', 'children'),
    Input('user', 'value'),
    Input('pw', 'value'),
    Input('locks', 'value'),
    Input('createBetButton', 'n_clicks'),
    Input('style', 'value'),
    Input('price', 'value'),
    Input('margin', 'value'),
    Input('sport', 'value'),
    Input('confirmButton', 'n_clicks'),
    Input('spread', 'value'),
    Input('line', 'value'),
    Input('overUnder', 'value'),
    Input('key', 'value')
)

def update(user, pw, locks, clicks, style, price, margin, sport, clicks2, spread, line, overUnder, key):
    status = checkStatus(user, pw, locks, clicks, style, price, margin, sport, clicks2)
    if key:
        openai.api_key = key
    else:
        return ["please enter a key", status]
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('python-sheets-2-357617-873449e7f558 (1).json', scope)
    client = gspread.authorize(creds)
    if not user:
        return [f"enter a username{clicks, clicks2}", status]
    if user in pwords:
        if not pw:
            return [f"enter a pw", status]
        elif pwords[user] == pw:
            if sport:
                try:
                    nums = [str(i) for i in range(10)]
                    sportsStr = ""
                    sportsNum = "0"
                    for i in sport:
                        if i not in nums:
                            sportsStr += i
                        else:
                            sportsNum += i
                    instance = get_sheet(sportsStr, int(sportsNum))
                except Exception as excep:
                    return [f"pls enter a valid sport{sport[:len(sport)-1], int(sport[-1])}\n{excep}", checkStatus(user, pw, locks,
                                                                                                                   clicks, style, price, margin, sport, clicks2)]
                if style:
                    if price:
                        if locks:
                            if clicks2 > len(sheetOutputs) and len(outputs) > 0:
                                try:
                                    sheetOutputs.append(outputs[-1])
                                    instance.update_cell(value=outputs[-1], row=12, col=10)
                                    return [f"bet entered successfully", status]
                                except Exception as excep:
                                    return [f"error exception1:\n{excep, outputs[-1]}", status]
                            inputTxt = locks
                            if clicks > len(inputs):
                                inputs.append(inputTxt)
                                try:
                                    if inputTxt[0] == "%":
                                        row = ""
                                        i = 1
                                        while inputTxt[i] != "&":
                                            row += inputTxt[i]
                                            i+=1
                                        col = str(inputTxt[i+1:])
                                        row = int(row)
                                        col = int(col)
                                        inputTxt = instance.cell(row=row, col=col).value
                                        locks = inputTxt
                                    outputTxt = simple_request(prompt=inputTxt,
                                                               model=style,
                                                               temperature=price,
                                                               max_tokens=margin,
                                                               top_p=spread,
                                                               frequency_penalty=line,
                                                               presence_penalty=overUnder
                                                               )
                                    outputs.append(outputTxt)
                                    return [f"lock: {locks}\n\nresult:\n{outputTxt}\nn_clicks={clicks}\n{price}", status]
                                except Exception as excep:
                                    return [f"error exception:\n{excep}", status]
                            else:
                                return ["pls submit ur bet", status]
                        else:
                            return [f"welcome to the holyland. pls enter a bet", status]
                    else:
                        return ["enter a valid amount", status]
                else:
                    return [f"pls enter a type of bet", status]
            else:
                return [f"pls enter a sport", status]
        else:
            return ["incorrect pw try again", status]
    else:
        return ["user not valid", status]


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)
