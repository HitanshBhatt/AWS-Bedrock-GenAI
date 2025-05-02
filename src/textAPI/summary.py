import boto3
import json

AWS_REGION_BEDROCK = "eu-central-1" #Make sure to set the region to the one you are using for Bedrock/where it is available

client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION_BEDROCK)

#Get the text and points (body) from the event and call the Titan model to summarize the text
def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text") #Text entry
    points = event["queryStringParameters"]["points"]   #Number of points in the 'queryStringParameters' of the event
    if text and points:
        titan_config = get_titan_config(text, points)   #Get configuration
        response = client.invoke_model(
            body=titan_config, 
            modelId="amazon.titan-text-express-v1", 
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        result = response_body.get("results")[0]
        return {
            "statusCode": 200,  #HTTP status code: OK
            "body": json.dumps({"summary": result.get("outputText")}),        
        }
    return {
        "statusCode": 400,  #HTTP status code: Bad Request
        "body": json.dumps({"error": "text and points required!"}),  
    }


#Function that builds the prompt for the Titan model
#text: the text to be summarized
#points: the number of points to summarize the text into
def get_titan_config(text: str, points: str):

    prompt = f"""Text: {text} \n
        From the text above, summarize the story in {points} points.\n    
    """

    return json.dumps(
    {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 4096,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1,
        },
    }
)