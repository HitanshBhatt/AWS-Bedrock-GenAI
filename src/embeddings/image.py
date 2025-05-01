import boto3
import json
import base64

from similarity import cosineSimilarity

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

#Test images to compare with the new image
images = [
    'images/1.png',
    'images/2.png',
    'images/3.png',
]

#Function to get the embedding for a given input image
def getImagesEmbedding(imagePath: str):
    #Load the image to be compared and encode it to base64
    with open(imagePath, "rb") as f:
        base_image = base64.b64encode(f.read()).decode("utf-8")

    #Invoke the model to get the embedding for the input image
    response = client.invoke_model(
        body=json.dumps({
            "inputImage": base_image,
        }), 
        modelId='amazon.titan-embed-image-v1',  #Using the Titan model for image embedding
        accept='application/json', 
        contentType='application/json')

    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')

#Placeholder list for the embeddings of the images
imagesWithEmbeddings = []

#Get the embedding for each image in the list of images
for image in images:
    imagesWithEmbeddings.append({
        'path': image,
        'embedding': getImagesEmbedding(image)
    })

test_image = 'images/cat.png'

#Get the embedding for the test image
test_image_embedding = getImagesEmbedding(test_image)

#List of similarities between the test image and the images in the list 
similarities = []

#Calculate the cosine similarity between the test image and each image in the list and append it to the similarities list
for image in imagesWithEmbeddings:
    similarities.append({
        'path': image['path'],
        'similarity': cosineSimilarity(image['embedding'], test_image_embedding)
    })

#Sort the similarities list by similarity in descending order
similarities.sort(key=lambda x: x['similarity'], reverse=True)

print(f"Similarities of '{test_image}' with:")
for similarity in similarities:
    print(f"  '{similarity['path']}': {similarity['similarity']:.2f}")