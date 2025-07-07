from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modal import App, Image, fastapi_endpoint, method
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/phi-2"

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
    def generate(self, prompt: str) -> str:
        print(f"Received prompt: {prompt}")

        age = 5

        formatted_prompt = (
            "You are a cheerful and safe assistant for children aged 3 to 10. "
            "You always use friendly, simple words, emojis, and never mention anything scary, violent, or adult. "
            f"Child (age {age}): {prompt}\n"
            "Assistant:"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        print(f"Generated response: {response}")
        return response

# web_app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

UNSAFE_KEYWORDS = [
    "violence", "death", "kill", "drugs", "sex", "blood", "die", "weapon",
    "gun", "hate", "stupid", "bomb", "explode", "murder", "attack", "fight",
    "war", "terrorist", "dead", "hurt"
]

def contains_inappropriate_content(text: str) -> bool:
    return any(word in text.lower() for word in UNSAFE_KEYWORDS)

phi_model = Phi2Model()

@app.function()
@fastapi_endpoint(method="POST", label='chat')
def chat(req: ChatRequest):
    if contains_inappropriate_content(req.prompt):
        return {"response": "Oops! Let's talk about something fun instead! 🌟"}
    
    phi_model.load.remote()
    result = phi_model.generate.remote(req.prompt)

    if contains_inappropriate_content(result):
        return {"response": "Let's talk about something fun instead! 🌈"}
    
    return {"response": result}