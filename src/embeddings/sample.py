import boto3
import json

#Using the Titan model to get embeddings for a text input
client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

#Fact and animal are the inputs to the model
fact = "The first moon landing was in 1969."
animal = "cat"

#Get the embedding for the fact
response = client.invoke_model(
    body=json.dumps(
        {
            "inputText": animal,
        }
    ),
    modelId="amazon.titan-embed-text-v1",
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
print(response_body.get("embedding"))   #Get the embedding for the fact
