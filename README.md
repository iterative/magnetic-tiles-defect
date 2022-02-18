# Saliency defect detection of magnetic tiles
### This project contains two components:
1. training of image segmentation model using DVC pipelines
2. web API that calls the above model on a provided image and responds with a binary segmentation mask

### The dataset is desribed in detail here: 
https://www.researchgate.net/profile/Congying-Qiu/publication/327701995_Saliency_defect_detection_of_magnetic_tiles/links/5b9fd1bd45851574f7d25019/Saliency-defect-detection-of-magnetic-tiles.pdf

### The dataset itself is hosted in this repository:
https://github.com/abin24/Magnetic-tile-defect-datasets. (including the dot)


## Prerequisites
- pipenv
## Setup
```bash
pipenv shell
pipenv install
echo "export PYTHONPATH=$PWD" >> $VIRTUAL_ENV/bin/activate
source $VIRTUAL_ENV/bin/activate
```

## Model Training
```bash
dvc repro
```

The model will be saved in the `models/` directory

## Web API serving (local)
```bash
uvicorn app.main:app
```

## Web API serving (Docker)
Build image
```bash
docker build . -t mag-tiles
```

Run container
```bash
docker run -p 8000:8000 -e PORT=8000 mag-tiles
```

## Test API 
With `curl`
```bash
curl -X POST -F 'image=@<PATH_TO_IMAGE>' -v http://127.0.0.1:8000/analyze
```

With python
```python
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import requests

url = 'http://127.0.0.1:8000/analyze'
file_path = Path(
    'data/MAGNETIC_TILE_SURFACE_DEFECTS/test_images/exp4_num_258590.jpg')
files = {'image': (str(file_path), open(file_path, 'rb'), "image/jpeg")}
response = requests.post(url, files=files)
data = json.loads(response.content)
pred = np.array(data['pred'])
plt.imsave(f'{file_path.stem}_mask.png', pred, cmap=cm.gray)
```

## Deploying to Heroku

```bash
heroku container:login
heroku create <APP_NAME>
heroku container:push --app <APP_NAME>
heroku container:release --app <APP_NAME>
```

Currently, the app is deployed to https://mag-tiles-api.herokuapp.com/analyze