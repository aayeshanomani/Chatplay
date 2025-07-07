Here's a concise and clear `README.md` for your **ChatPlay** model:

---

# 🎈 ChatPlay

**ChatPlay** is a cheerful and safe AI chatbot designed for children aged 3 to 10. It runs on [Modal](https://modal.com/) using the `microsoft/phi-2` language model and provides friendly, paragraph-style responses using simple words and emojis. It filters unsafe content and is designed to be deployable as a serverless API endpoint.

---

## 🚀 Features

* ✅ Hosted on [Modal](https://modal.com/)
* ✅ GPU-backed inference with Phi-2 (`microsoft/phi-2`)
* ✅ FastAPI backend exposed as a Modal HTTP endpoint
* ✅ Child-safe prompt filtering (blocks adult/violent language)
* ✅ Friendly assistant tone with emoji support
* ✅ Paragraph-style, conversational replies

---

## 🧠 Prompt Design

Every prompt is wrapped like this to ensure child-friendly behavior:

```python
"You are a cheerful and safe assistant for children aged 3 to 10. "
"You always use friendly, simple words, emojis, and never mention anything scary, violent, or adult. "
f"Child (age {age}): {user_input}\n"
"Assistant:"
```

---

## 🛠️ Installation (Local Dev)

```bash
git clone https://github.com/your-username/chatplay.git
cd chatplay
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Usage

### 🌐 Deploy via Modal

```bash
modal serve main.py
```

This creates an endpoint like:

```bash
https://your-username--chat-dev.modal.run
```

### 🔁 Example Request (Curl)

```bash
curl -X POST https://your-username--chat-dev.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a fun bedtime story!"}'
```

---

## 🛡️ Safety Filter

The backend blocks prompts containing:

```
["violence", "death", "kill", "drugs", "sex", "blood", "die", "weapon"]
```

Returns HTTP 400 if unsafe input is detected.

---

## 📁 Files

| File               | Description                           |
| ------------------ | ------------------------------------- |
| `main.py`          | Modal app + FastAPI logic             |
| `requirements.txt` | Python deps for Modal + Transformers  |
| `.venv/`           | Virtual environment (locally ignored) |

---

## 🧩 TODO

* [ ] Add persistent session support
* [ ] Include age-based tone adjustments
* [ ] Build a frontend (chat UI)

---

## ✨ Credits

* Built using [Modal](https://modal.com/)
* Powered by [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
* Designed by Aayesha Nomani

---

Let me know if you'd like a version with badges, frontend setup, or deployment guide to Hugging Face/Render too.
