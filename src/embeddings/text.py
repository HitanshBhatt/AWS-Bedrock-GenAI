import boto3
import json

from similarity import cosineSimilarity
client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

#List of facts to compare with the new fact
facts = [
    'The first computer was invented in the 1940s.',
    'John F. Kennedy was the 35th President of the United States.',
    'The first moon landing was in 1969.',
    'The capital of France is Paris.',
    'Earth is the third planet from the sun.',
]

#New fact to compare with the list of facts
newFact = 'I like to play computer games'
question = 'Who is the president of USA?'

#Function to get the embedding for a given input text
def getEmbedding(input: str):
    #Invoke the model to get the embedding for the input text
    response = client.invoke_model(
        body=json.dumps({
            "inputText": input,
        }), 
        modelId='amazon.titan-embed-text-v1', 
        accept='application/json', 
        contentType='application/json')

    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')

#Placeholder list for the embeddings of the facts
factsWithEmbeddings = []

for fact in facts:
    factsWithEmbeddings.append({
        'text': fact,
        'embedding': getEmbedding(fact)
    })

#Calculate the embedding for the new fact
newFactEmbedding = getEmbedding(question)

#List of similarities between the new fact and the facts in the list
similarities = []

#Calculate the cosine similarity between the new fact and each fact in the list and append it to the similarities list
for fact in factsWithEmbeddings:
    similarities.append({
        'text': fact['text'],
        'similarity': cosineSimilarity(fact['embedding'], newFactEmbedding)
    })

print(f"Similarities for fact: '{question}' with:")
similarities.sort(key=lambda x: x['similarity'], reverse=True)  #Sort by similarity in descending order
for similarity in similarities:
    print(f"  '{similarity['text']}': {similarity['similarity']:.2f}")

