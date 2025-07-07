from fastapi import FastAPI
from pydantic import BaseModel
from modal import App, Image, fastapi_endpoint, method
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"

UNSAFE_KEYWORDS = [
    "violence", "death", "kill", "drugs", "sex", "blood", "die", "weapon",
    "gun", "hate", "stupid", "bomb", "explode", "murder", "attack", "fight",
    "war", "terrorist", "dead", "hurt"
]

image = (
    Image.debian_slim()
    .pip_install("torch", "transformers", "fastapi", "uvicorn", "numpy<2")
)

# Define Modal App
app = App("chatplay-web", image=image)

@app.cls(gpu="A10G", image=image)
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
        output = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        print(f"Generated response: {response}")
        return response
    
phi_model = Phi2Model()

def is_unsafe(text: str) -> bool:
    return any(word in text.lower() for word in UNSAFE_KEYWORDS)

def safe_generate_response(message: str, age: int) -> str:
    if is_unsafe(message):
        return {"response": "Oops! Let's talk about something fun instead! 🌟"}
    
    phi_model.load.remote()
    result = phi_model.generate.remote(message, age)
    if is_unsafe(result):
        return "Let's talk about something fun instead! 🌈"
    return result
    
fastapi_app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    age: int

@fastapi_app.post("/chat")
def chat_endpoint(req: ChatRequest):
    reply = safe_generate_response(req.message, req.age)
    return {"reply": reply}

@app.function()
@fastapi_endpoint(method="POST")
def chat(req: ChatRequest):
    return chat_endpoint(req)