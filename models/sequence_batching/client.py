from tritonclient.utils import np_to_triton_dtype
import numpy as np
import tritonclient.http as httpclient
from tritonclient import grpc as grpcclient
from tritonclient import utils as utils
import json
import time

def sequence_batching_client():
    url = "localhost:8000" # Use http client
    model_name = "sequence_batching"
    sequence = np.array([1,2,3,4], dtype=np.int32)
    # sequence = np.expand_dims(sequence, axis=0)
    
    # Create the inference server client
    try:
        triton_client = httpclient.InferenceServerClient(url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        return
    
    # Sanity check
    metadata = triton_client.get_model_metadata(model_name)
    print(metadata)
    
    model_config = triton_client.get_model_config(model_name)

    responses = []

    for i in range(len(sequence)):
        inputs = [httpclient.InferInput("my_input", [1, 1], np_to_triton_dtype(sequence.dtype))]
        inputs[0].set_data_from_numpy(sequence[np.newaxis, i:i+1], binary_data=True)
        outputs = [httpclient.InferInput("my_output", [1, 1], np_to_triton_dtype(sequence.dtype))]

        response = triton_client.async_infer(
            model_name,
            inputs,
            outputs=outputs,
            sequence_id=1,
            sequence_start=(i==0),
            sequence_end=(i==len(sequence)-1),
        )

        responses.append(response)
    
    for response in responses:
        start_time = time.time()
        result = response.get_result()
        response_get_time = time.time() - start_time
        # Print the results
        print("Response time:", response_get_time)
        print(result.get_output("my_output"))


if __name__ == "__main__":
    sequence_batching_client()
