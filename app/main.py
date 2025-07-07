from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import uvicorn
from typing import List, Dict
from services.ConversationSuggestionEngine import ConversationSuggestionEngine
import json

app = FastAPI(title="Conversation Suggestions API")

suggestion_engine = ConversationSuggestionEngine(model_name="llama2")


# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_histories: Dict[str, List[Dict[str, str]]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.conversation_histories[client_id] = []

    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.remove(websocket)
        if client_id in self.conversation_histories:
            del self.conversation_histories[client_id]

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    def add_message_to_history(self, client_id: str, role: str, content: str):
        if client_id not in self.conversation_histories:
            self.conversation_histories[client_id] = []

        self.conversation_histories[client_id].append(
            {"role": role, "content": content}
        )

        # Keep only last 20 messages
        if len(self.conversation_histories[client_id]) > 20:
            self.conversation_histories[client_id] = self.conversation_histories[
                client_id
            ][-20:]


manager = ConnectionManager()


@app.get("/")
async def get_homepage():
    """Serve a simple HTML page for testing WebSocket connection."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Conversation Suggestions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .chat-box { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
            .message { margin-bottom: 10px; padding: 5px; border-radius: 5px; }
            .user-message { background-color: #e3f2fd; }
            .ai-message { background-color: #f3e5f5; }
            .suggestions { background-color: #e8f5e8; }
            .input-section { display: flex; gap: 10px; margin-bottom: 10px; }
            #messageInput { flex: 1; padding: 10px; }
            button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .suggestion-item { 
                background-color: #f8f9fa; 
                border: 1px solid #dee2e6; 
                padding: 8px; 
                margin: 5px 0; 
                border-radius: 4px; 
                cursor: pointer; 
            }
            .suggestion-item:hover { background-color: #e9ecef; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Conversation Suggestions</h1>
            <div id="chatBox" class="chat-box"></div>
            <div class="input-section">
                <input type="text" id="messageInput" placeholder="Type your message...">
                <button onclick="sendMessage()">Send</button>
                <button onclick="getSuggestions()">Get Suggestions</button>
            </div>
            <div id="suggestions"></div>
        </div>

        <script>
            const ws = new WebSocket("ws://localhost:8000/ws/test-client");
            const chatBox = document.getElementById("chatBox");
            const messageInput = document.getElementById("messageInput");
            const suggestionsDiv = document.getElementById("suggestions");

            ws.onopen = function(event) {
                addMessage("Connected to server", "system");
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === "suggestions") {
                    displaySuggestions(data.suggestions);
                } else if (data.type === "message") {
                    addMessage(data.content, data.role);
                }
            };

            function addMessage(content, role) {
                const messageDiv = document.createElement("div");
                messageDiv.className = `message ${role}-message`;
                messageDiv.textContent = `${role}: ${content}`;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function displaySuggestions(suggestions) {
                suggestionsDiv.innerHTML = "<h3>Suggestions:</h3>";
                suggestions.forEach((suggestion, index) => {
                    const suggestionDiv = document.createElement("div");
                    suggestionDiv.className = "suggestion-item";
                    suggestionDiv.textContent = `${index + 1}. ${suggestion}`;
                    suggestionDiv.onclick = () => {
                        messageInput.value = suggestion;
                    };
                    suggestionsDiv.appendChild(suggestionDiv);
                });
            }

            function sendMessage() {
                const message = messageInput.value;
                if (message.trim()) {
                    ws.send(JSON.stringify({
                        type: "message",
                        content: message
                    }));
                    messageInput.value = "";
                }
            }

            function getSuggestions() {
                ws.send(JSON.stringify({
                    type: "get_suggestions"
                }));
            }

            messageInput.addEventListener("keypress", function(e) {
                if (e.key === "Enter") {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Wait for message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            if message_data["type"] == "message":
                # Add user message to history
                user_message = message_data["content"]
                manager.add_message_to_history(client_id, "user", user_message)

                # Echo the message back (you can replace this with actual AI response)
                await manager.send_message(
                    {"type": "message", "role": "user", "content": user_message},
                    websocket,
                )

                # Generate AI response (placeholder)
                avatar_response = f"I understand you said: {user_message}"
                manager.add_message_to_history(client_id, "assistant", avatar_response)

                await manager.send_message(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": avatar_response,
                    },
                    websocket,
                )
            elif message_data["type"] == "get_suggestions":
                # Get conversation history for this client
                conversation_history = manager.conversation_histories.get(client_id, [])

                # Generate suggestions asynchronously
                suggestions = await asyncio.get_event_loop().run_in_executor(
                    None,
                    suggestion_engine.get_contextual_suggestions,
                    conversation_history,
                    "",  # Topic
                    None,  # avatar_preferences
                )

                # Send suggestions back to client
                await manager.send_message(
                    {"type": "suggestions", "suggestions": suggestions}, websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
        print(f"Client {client_id} disconnected")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Conversation suggestions API is operational.",
    }


@app.post("/suggestions")
async def get_suggestions_api(request: dict):
    """REST API endpoint for getting suggestions

    Args:
        request (dict): dictionary containing conversation_history, topic, & avatar_settings
    """
    conversation_history = request.get("conversation_history", [])
    topic = request.get("topic", [])
    user_interests = request.get("user_interests", [])

    suggestions = suggestion_engine.get_contextual_suggestions(
        recent_messages=conversation_history, topic=topic, user_interests=user_interests
    )

    return {"suggestions": suggestions, "status": "success"}


if __name__ == "__main__":
    print("Starting Conversation Suggestions API...")
    print("Visit http://localhost:8000 for the web interface")
    print("WebSocket endpoint: ws://localhost:8000/ws/{client_id}")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
