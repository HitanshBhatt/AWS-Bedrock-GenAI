from langchain_aws import BedrockLLM as Bedrock #Build a Bedrock client
from langchain_core.prompts import ChatPromptTemplate   #Import the ChatPromptTemplate class from langchain_core.prompts
import boto3

AWS_REGION = "us-west-2"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

#Bedrock model from langhchain_aws
model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)

#Invoke the model with a simple prompt
def invoke_model():
    response = model.invoke("What is the highest mountain in the world?")
    print(response)

#Create a chain using the model and a prompt template
def first_chain():
    #Multiple message that we can pass as a prompt to the model
    #The first message is a system message that tells the model what to do
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short description for the product provided by the user",
            ),
            ("human", "{product_name}"),    #Human message that will be replaced by the user input; Product name is a placeholder
        ]
    )
    chain = template.pipe(model)    #Pipe the model to the template to create a chain

    response = chain.invoke({"product_name": "bicycle"})    #Invoke the chain with a product name
    print(response)


first_chain()
