const API_URL = "http://127.0.0.1:8000/chat";

const chatBox = document.getElementById("chat-box");
const input = document.getElementById("user-input");

function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.innerText = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    const data = await response.json();

    addMessage(
      `${data.response}\n\n[Emotion: ${data.emotion}, Confidence: ${data.confidence}]`,
      "bot"
    );

  } catch (err) {
    addMessage("❌ Server error. Is backend running?", "bot");
  }
}

input.addEventListener("keypress", function (e) {
  if (e.key === "Enter") sendMessage();
});
