import boto3
import json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = runtime.invoke_endpoint(
    EndpointName="qna-endpoint",  
    ContentType="application/json",
    Body=json.dumps({"question": "What is the capital of India?"})
)

result = json.loads(response["Body"].read())
print(result)
