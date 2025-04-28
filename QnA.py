import boto3
import json

def ask_bedrock(question):
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1') 
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": 300
    }

    response = bedrock_runtime.invoke_model(
        body=json.dumps(body),
        modelId='anthropic.claude-3-haiku-20240307-v1:0', 
        accept='application/json',
        contentType='application/json'
    )

    result_json = json.loads(response['body'].read().decode('utf-8'))  
    answer_text = result_json['content'][0]['text']  
    return answer_text

if __name__ == "__main__":
    question = "What is the capital of India?"
    answer = ask_bedrock(question)
    print("Answer from Bedrock:", answer)
