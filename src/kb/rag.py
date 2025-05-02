import boto3
import json

AWS_REGION_BEDROCK = "us-west-2"

client = boto3.client(
    service_name="bedrock-agent-runtime", region_name=AWS_REGION_BEDROCK
)

def handler(event, context):
    body = json.loads(event["body"])
    question = body.get("question")
    if question:
        response = client.retrieve_and_generate(
            input={"text": question},   #Input type is text
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "PSX4JIJMQN",    #Get this from the console
                    "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-v2",  #arn:aws:serviceUsed:region::foundation-model/modelID
                    #modelID can be found in the 'API Request' section of the model page in the console
                },
            },
        )
        #Extract the answer from the response
        answer = response.get("output").get("text")
        return {
            "statusCode": 200,  #HTTP status code: OK
            "body": json.dumps({"answer": answer}),
        }
    return {
            "statusCode": 400,  #HTTP status code: Bad Request
            "body": json.dumps({"error": "question needed"}),
        }


