import boto3
import json

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

#Pass context as a list of strings to maintain the conversation history
def get_configuration(context_list:str):
    prompt = "\n".join(context_list)    #Combine the context list into a single string
    return json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 4096,
                "stopSequences": [],
                "temperature": 0.7, # Adjusted temperature for more varied/human-like responses
                "topP": 1
            }
    })

#Initialize the conversation with a system-style greeting
context_list = ["Bot: Hello! I am a chatbot. I can help you with anything you want to talk about."]

print(context_list[0])  #Display the starting message to the user

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    context_list.append(f"User: {user_input}")  # Append user input to the context list

    response = client.invoke_model(
        body=get_configuration(user_input), 
        modelId="amazon.titan-text-express-v1", 
        accept="application/json", 
        contentType="application/json")
    
    response_body = json.loads(response.get('body').read())
    bot_response = response_body.get('results')[0].get('outputText')    #Extract the bot's response

    print(f"Bot: {bot_response}")  #Display the bot's response to the user
    context_list.append(f"Bot: {bot_response}")  # Append bot response to the context list for history
