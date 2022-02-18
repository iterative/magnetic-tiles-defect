import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import requests

url = 'https://mag-tiles-api.herokuapp.com/analyze'
file_path = Path(
    'data/MAGNETIC_TILE_SURFACE_DEFECTS/test_images/exp4_num_258590.jpg')
files = {'image': (str(file_path), open(file_path, 'rb'), "image/jpeg")}
response = requests.post(url, files=files)
data = json.loads(response.content)
pred = np.array(data['pred'])
plt.imsave(f'{file_path.stem}_mask.png', pred, cmap=cm.gray)
