import boto3
import json
import pprint

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

titan_model_id = 'amazon.titan-text-express-v1' #Titan model id from AWS Bedrock

#Dictionary to configure the model
# The input text is the prompt for the model, and the text generation config specifies the parameters for text generation.
titan_config = json.dumps({
            "inputText": "Tell me a story about a dragon",  #Input text for the model
            #Eveerything below is adapted from the 'API Request' section from Bedrock console
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1   #camelCase as per Titan API
            }
        })

llama_model_id = "meta.llama2-13b-chat-v1"
llama_config = json.dumps({
    "prompt": "Tell me a story about a dragon",
    "max_gen_len": 512,
    "temperature": 0,
    "top_p": 0.9,
})

#The client invokes the model with the specified configuration and model ID.
response = client.invoke_model(
    body=llama_config,
    modelId=llama_model_id,
    accept="application/json",  #Value from 'API Request' section from Bedrock console
    contentType="application/json"
)

#The response is read and parsed as JSON.
# The response contains the generated text from the model.
response_body = json.loads(response.get('body').read())

pp = pprint.PrettyPrinter(depth=4)
# pp.pprint(response_body.get('results')) # titan config
pp.pprint(response_body["generation"]) # llama config - 'generation' is the key for the generated text