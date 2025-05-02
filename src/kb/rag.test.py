from rag import handler
import json

event = {
    "body": json.dumps({"question": "What does GDPR stand for?"})   #Example question
}

response = handler(event, {})

print(response)