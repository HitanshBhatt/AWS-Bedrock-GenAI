import boto3
import json
import base64

client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2")

#Prompt for the image generation model
stability_image_config = json.dumps({
    "text_prompts": [
        {
            "text": 'a photo of a dragon',
        }
    ],
    "height": 512,
    "width": 512,
    "cfg_scale": 10,
    "style_preset": '3d-model',
})

#Invoking the model with the specified configuration and model ID.
response = client.invoke_model(
    body=stability_image_config, 
    modelId="stability.stable-diffusion-xl-v1", 
    accept="application/json", 
    contentType="application/json")

response_body = json.loads(response.get("body").read())
#Image is in the base64 format in the response body.
base64_image = response_body.get("artifacts")[0].get("base64")  #Response varies based on the model used

base_64_image = base64.b64decode(base64_image)

file_path = "dragon.png"
with open(file_path, "wb") as f:
    f.write(base_64_image)
