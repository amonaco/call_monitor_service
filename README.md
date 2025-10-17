### call_monitor_service

Service that analyzes telephone call prefixes, can learn and find abnormalities as a simple yet robust call fraud-detection system

This could be used either to process CDR (call detail records) or live calls (say from Asterisk or Freeswitch). It learns (when in learning-mode) from call prefixes and finds abnormalities (when running on policing-mode). It uses PyTorch and runs on FastAPI.

### Status
While fully functional and created to address a personal, real-world need, this project is still under active development and should be considered a work in progress (WIP).

* Still need to make auth consistent
* Config file via dotenv

### Install

`pip install fastapi uvicorn torch numpy`

### Run

`uvicorn call_monitor_service:app --reload`

### Usage

1. Check status

`curl http://127.0.0.1:8000/status`

2. Feed prefixes (learning mode)

```
curl -X POST -H "Content-Type: application/json" \
     -d '{"prefix":"+1"}' http://127.0.0.1:8000/feed
```

3. Switch to policing mode
```
curl -X POST -H "Content-Type: application/json" \
     -d '{"auth_key":"supersecret123","mode":"policing"}' \
     http://127.0.0.1:8000/admin/mode
```

4. Feed new data (now checks anomalies)
```
curl -X POST -H "Content-Type: application/json" \
     -d '{"prefix":"+91"}' http://127.0.0.1:8000/feed
```
5. Forget (TODO add auth)
```
curl -X POST http://localhost:5000/forget \
  -H "Content-Type: application/json" \
  -d '{"prefixes": ["+49", "+31"]}'
  ```
