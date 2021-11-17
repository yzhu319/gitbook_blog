---
description: full repo link--
---

# Serve a ML model with a Dockerized web server

Now that we have went a long way to train a fantastic model, but how to serve it? In this article I briefly explains&#x20;

1. how to code a web server using FastAPI&#x20;
2. how to Dockerize the server.&#x20;

An example of the trained model is a gradient-boosting-tree classifier, it contains all the parameters to be used in inferencing. The model file 'lightgbm\_model.txt' contains all the saved parameters, which will be used in the web server.



### Coding the server

Notice that the server's code must be in the file `main.py` within a directory called `app`, following FastAPI's guidelines.

Here I use `lgb.booster()` function for loading the pre-trained model saved in the `app/lightgbm_model.txt` file, `numpy` for tensor manipulation, and the rest for developing the web server with `FastAPI`.

Also, create an instance of the `FastAPI` class. This instance will handle all of the functionalities for the server:

```python
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Predicting Flight")
```



Put things together with the 'prediction' method, here is the main.py inside the app/ directory



```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from pydantic import BaseModel, conlist
from typing import List

# Server
import uvicorn
from fastapi import FastAPI
app = FastAPI(title="Predicting booking likelihood")

print("begin loading model\n")
model_filename = "/app/lightgbm_model.txt"
bst = lgb.Booster(model_file= model_filename)
print("model file loaded\n")

class Data(BaseModel):
    batches: List[conlist(item_type=float, min_items=11, max_items=11)]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(data: Data):

    # Extract data
    batches = data.batches
    np_batches = np.array(batches)

    # Create and return prediction
    # predict method: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html
    prediction_logits = bst.predict(np_batches)

    from scipy.special import softmax
    # calculate softmax
    prediction = softmax(prediction_logits).tolist()

    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5051)

# make inference with input Data
# curl -X POST http://127.0.0.1:5051/predict -d @./shop_examples_json/qid_3999.json -H "Content-Type: application/json"

```

Now the server's code is ready for inference, launch the server locally by using the command `uvicorn main:app --reload` while on the same directory as the `main.py` file.

### Dockerizing the server

Going forward all commands are run within the `serve_API_withBatch_Docker/` directory.

Create a directory called `app` and place `main.py` (the server) and its dependencies (`lightgbm_model.txt`) there as explained on the official FastAPI [docs](https://fastapi.tiangolo.com/deployment/docker/) on how to deploy with Docker. This should result in a directory structure that looks like this:

```
..
└── serve_API_withBatch_Docker
    ├── app/
    │   ├── main.py (server code)
    │   └── lightgbm_model.txt (serialized classifier)
    ├── requirements.txt (Python dependencies, optional)
    ├── shop_examples_json/ (examples to test the server)
    ├── README.md (this file)
    └── Dockerfile
```

### Create the Dockerfile

The `Dockerfile` is made up of all the instructions required to build an image.

### Build the image

Now that the `Dockerfile` is ready, it is time to build the image. To do so, use the `docker build` command.

```bash
docker build -t flight_app:batch .
```

One can use the `-t` flag to specify the name of the image and its tag. The tag comes after the colon so in this case the name is `flight_app` and the tag is `batch`.

### Run the container

Now that the image has been successfully built (after a few minutes) it is time to run a container out of it. Use the following command:

```bash
docker run --rm -p 80:80 flight_app:batch
```

At the end of the command is the name and tag of the image we want to run.

After some seconds the container will start and spin up the server within. One should be able to see FastAPI's logs being printed in the terminal.

Now head over to [localhost:80](http://localhost) and should see a message about the server spinning up correctly.

### Make requests to the server

Now that the server is listening to requests on port 80, one can send `POST` requests to it for prediction.

* Use `curl` to make `POST` requests to get predictions from servers.

```bash
curl -X POST http://localhost:80/predict \
        -d @./shop_examples_json/qid_3999.json \
        -H "Content-Type: application/json"
```

* Use FastAPI Interface to make `POST` requests to get predictions from servers.

Start the webserver by: python main.py Then go to http://localhost:80/docs, click 'try now' to test with JSON inputs



