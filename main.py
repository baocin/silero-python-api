import torch
import zipfile
import torchaudio
from glob import glob
from flask import Flask
from flask import request

app = Flask(__name__)


device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
model, decoder, utils = torch.hub.load(repo_or_dir='./models/',
                                       source='local',
                                       model='silero_stt',
                                       language='en',
                                       device=device)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils  # see function signature for details

@app.route('/')
@app.route('/index')
def index():
    return 'Hello world!'

@app.route('/s2t')
def speechToText():
    # Must be WAV file (48kHz preferred)
    url = request.args.get('url')
    #Example: https://opus-codec.org/static/examples/samples/speech_orig.wav
    #Example 2: https://s3.us-east-2.amazonaws.com/steele.red/thisisatest.wav
    # http://localhost:5000/s2t?url=https://s3.us-east-2.amazonaws.com/steele.red/thisisatest.wav

    print("Processing:", url)
    torch.hub.download_url_to_file(url, dst ='speech_orig.wav', progress=True)
    test_files = glob('speech_orig.wav') 
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)
    output = model(input)
    outputString = "";
    for example in output:
        print(decoder(example.cpu()))
        outputString += decoder(example.cpu())
    return outputString


app.config['DEBUG'] = False
if __name__ == "__main__":
    # only enable debug when calling python directly during dev
    # gunicorn will bypass this when it is called in a container
    app.config['DEBUG'] = True
    app.run(host='0.0.0.0')