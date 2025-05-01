from langchain_aws import BedrockLLM as Bedrock
from langchain_aws import BedrockEmbeddings #Embedding model from langchain_aws
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS  #Vector database from Facebook
import boto3

#List of strings to be used as data
my_data = [
    "The weather is nice today.",
    "Last night's game ended in a tie.",
    "Don likes to eat pizza.",
    "Don likes to eat pasta.",
]

question = "What does Don like to eat?"

AWS_REGION = "us-west-2"

#Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

#Model for langchain
model = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock)    #LLM model from langchain_aws

#Embedding model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

# create vector store
vector_store = FAISS.from_texts(my_data, bedrock_embeddings)

# create retriever to retieve information from the vector store
retriever = vector_store.as_retriever(
    #k = 2, give me the top 2 results
    search_kwargs={"k": 2}  # maybe we can add a score threshold here?
)

results = retriever.invoke(question)

#Transform results into a string
results_string = []
for result in results:
    results_string.append(result.page_content)

# build chat template:
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}"),    #Input is our question
    ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response)








