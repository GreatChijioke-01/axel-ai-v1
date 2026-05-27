const textInput = document.getElementById('textInput');
const chatbox = document.getElementById('chatbox');
const sendBtn = document.getElementById('sendBtn');

async function getResponse(){
    const userText = textInput.value.trim();
    if(userText === "") return;

    // Display user message
    appendMessage(userText, "user-msg");
    textInput.value = "";

    // Fetch from Flask
    try {
        const response = await fetch("/get", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: userText})
        });

        const data = await response.json();

        // Display Axel's response
        appendMessage(data.response, "axel-msg");
    } catch (error) {
        appendMessage("System Error: Could not reach Axel.", "axel-msg");  
    }
}

function appendMessage(text, className){
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${className}`;
    msgDiv.innerHTML = `<p>${text}</p>`;
    chatbox.appendChild(msgDiv);

    // Auto-scroll to the latest message
    chatbox.scrollTop = chatbox.scrollHeight;
}

// Event listeners
sendBtn.addEventListener("click", getResponse);

textInput.addEventListener("keypress", function(e){
    if(e.key === "Enter"){
        getResponse();
    }
});