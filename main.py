import re
from fastapi import FastAPI
from pydantic import BaseModel
from modal import App, Image, fastapi_endpoint, method
from better_profanity import profanity
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"

UNSAFE_KEYWORDS = [
    "violence", "death", "kill", "drugs", "sex", "blood", "die", "weapon",
    "gun", "hate", "stupid", "bomb", "explode", "murder", "attack", "fight",
    "war", "terrorist", "dead", "hurt", "naked", "porn", "abuse", "touch",
    "molest", "private part", "rape"
]

SUSPICIOUS_KEYWORDS = [
    "hide", "secret", "don’t tell", "lie", "run away",
    "hate parents", "sneak", "trick", "keep from parents", "don’t show",
    "don't show", "escape", "cheat"
]

SUSPICIOUS_PATTERNS = [
    "how to .* from parents",
    "how to .* a secret",
    "how to .* without telling",
    "ways to .* secretly",
    "can I .* without my mom",
    "how do I .* lie",
]

image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "fastapi", "uvicorn", "numpy<2", "better_profanity")
)

# Define Modal App
app = App("chatplay-web", image=image)

@app.cls(gpu="A10G", image=image, min_containers=1)
class Phi2Model:
    @method()
    def load(self):
        print("Loading model...")

        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).cuda()

        print("Model loaded successfully.")

        try:
            inputs = self.tokenizer("Hello", return_tensors="pt").to("cuda")
            _ = self.model.generate(**inputs, max_new_tokens=5)
            print("Warm-up generation complete.")
        except Exception as e:
            print(f"Warm-up failed: {e}")

    @method()
    def generate(self, user_input: str, age: int) -> str:
        print(f"Received prompt: {user_input}")

        formatted_prompt = (
            "You are a cheerful and safe assistant for children aged 3 to 10. "
            "You always use friendly, simple words, emojis, and never mention anything scary, violent, or adult. "
            f"Child (age {age}): {user_input}\n"
            "Assistant:"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.95,
            do_sample=True
        )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        if f"Child (age {age}):" in response:
            response = response.split(f"Child (age {age}):")[-1].strip()

        print(f"Generated response: {response}")
        return response
    
phi_model = Phi2Model()
profanity.load_censor_words()

def is_unsafe(text: str) -> bool:
    if profanity.contains_profanity(text):
        return True
    return any(word in text.lower() for word in UNSAFE_KEYWORDS)

def is_suspicious(text: str) -> bool:
    lowered = text.lower()
    
    # Keyword match
    if any(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS):
        return True
    
    # Pattern match
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lowered):
            return True

    return False

def safe_generate_response(message: str, age: int) -> str:
    if is_unsafe(message):
        return "That sounds like something important. I think it's best to talk to your parents or a grown-up you trust about it. I'm here for fun stories, games, and learning! 😊"
    
    '''
    TODO: Improve the message

    Example case:
    curl -X POST https://aayeshanomani--chatplay-web-chat.modal.run \
    -H "Content-Type: application/json" \
    -d '{"message": "today i got bullied in class", "age": 7}'

    {"reply":"I'm here to help with fun stories and games, not secrets! 😊 Let's play something else?"}
    '''
    
    if is_suspicious(message):
        return "I'm here to help with fun stories and games, not secrets! 😊 Let's play something else?"

    phi_model.load.remote()
    response = phi_model.generate.remote(message, age)

    return response
    
fastapi_app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    age: int

@fastapi_app.post("/chat")
def chat_endpoint(req: ChatRequest):
    reply = safe_generate_response(req.message, req.age)
    return {"reply": reply}

@app.function(min_containers=1)
@fastapi_endpoint(method="POST")
def chat(req: ChatRequest):
    return chat_endpoint(req)