# Machine Backend

## Setup

Create new python environment
```
python3 -m venv .venv
```

Activate python environment
```
. .venv/bin/activate
```

Install package.
```
pip install -e .
```

## Catwell

### Start Backend locally

```
. .venv/bin/activate
python machine_backend/app_catwell.py
```

### Endpoints

#### Image
Get landscape as image (only for debugging)

```
http://0.0.0.0:8090/image?size=500&experiment-name=test&seed=42
```

#### Evaluate Landscape

Evaluate the landscape at a given point

```
POST http://0.0.0.0:8090/eval
content-type: application/json

{
  "requestId": "asdasdasd",
  "data": {
    "action": {
      "x": 0.2,
      "y": 0.3
    },
    "environment": {
      "type": "catwell",
      "version": "foo",
      "experimentName": "test",
      "seed": 1
    }
  }
}

```

