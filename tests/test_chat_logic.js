// Test Script for Prompt Logic
// Usage: node tests/test_chat_logic.js

// Mock Data simulating Qdrant Output (Array of items)
const mockQdrantItems = [
    {
        json: {
            payload: {
                text_content: "נשים ועבדים וקטנים פטורין מקריאת שמע...",
                metadata: { masechet: "Berakhot", daf: "20a" }
            }
        }
    },
    {
        json: {
            payload: {
                text_content: "תנו רבנן ברוך שם כבוד מלכותו לעולם ועד...",
                metadata: { masechet: "Berakhot", daf: "13a" }
            }
        }
    }
];

// Mock Telegram Input
const mockTelegramInput = {
    item: {
        json: {
            message: {
                text: "האם נשים חייבות בקריאת שמע?"
            }
        }
    }
};

// Simulation of n8n environment
const $ = (nodeName) => {
    if (nodeName === 'Telegram Trigger') return mockTelegramInput;
    return {};
};
const items = mockQdrantItems;

// --- START LOGIC COPY (from prompt_assembly_logic.js) ---
const userQuestion = $('Telegram Trigger').item.json.message.text;

const sources = items.map((item, index) => {
    const payload = item.json.payload || item.json; 
    const text = payload.text_content || payload.content || "";
    const meta = payload.metadata || {};
    const citation = `[${meta.masechet || "Source"} ${meta.daf || index + 1}]`;
    
    return `${citation}\n${text}`;
}).join("\n\n--------------------\n\n");

const systemPrompt = `You are a Talmud study assistant.
You must answer ONLY using the sources below.
If the answer is not explicitly stated in the provided text, reply EXACTLY: "אין מקור מפורש".
Do not use outside knowledge.
Cite your sources using the format [Masechet Daf].

Sources:
${sources}
`;

const result = {
    system_prompt: systemPrompt,
    user_prompt: `Question: ${userQuestion}`
};
// --- END LOGIC COPY ---

console.log("=== Generated System Prompt ===");
console.log(result.system_prompt);
console.log("\n=== Generated User Prompt ===");
console.log(result.user_prompt);
