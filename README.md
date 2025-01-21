# Dyanamic Sequence Batcher Testing

Start the server in one terminal:

```
docker run -w /workspace/python_backend/ -v $PWD:/workspace/python_backend/ --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository models
```

Then run the client in another terminal (make sure that the server is
fully started first. You need to see "Started Metrics Service at
0.0.0.0:8002"):

```
docker run -w /workspace/python_backend/ -v $PWD:/workspace/python_backend/ -ti --net host nvcr.io/nvidia/tritonserver:24.09-py3-sdk python models/sequence_batching/client.py
```

Notice the following in
models/sequence_batching/config.pbtxt. "max_batch_size" is 2, but
max_queue_delay_microseconds is 2 seconds, so we expect that the
max_queue_delay will timeout every time in this workload.

The expected output from the client is:

```
Response time: 1.9610955715179443
{'name': 'my_output', 'datatype': 'INT32', 'shape': [1, 1], 'data': [1]}
Response time: 2.00224232673645
{'name': 'my_output', 'datatype': 'INT32', 'shape': [1, 1], 'data': [2]}
Response time: 2.001756429672241
{'name': 'my_output', 'datatype': 'INT32', 'shape': [1, 1], 'data': [3]}
Response time: 2.002234697341919
{'name': 'my_output', 'datatype': 'INT32', 'shape': [1, 1], 'data': [4]}
```

If the dynamic sequence batcher were allowed to put multiple elements
of the same sequence into the same batch, the response times would be
almost 0. So it appears that dynamic sequence batching is working
correctly.