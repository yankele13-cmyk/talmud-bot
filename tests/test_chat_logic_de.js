// Test Script for German Prompt Logic
// Usage: node tests/test_chat_logic_de.js

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

// Mock Telegram Input (German question)
const mockTelegramInput = {
    item: {
        json: {
            message: {
                text: "Sind Frauen verpflichtet, das Schma zu lesen?"
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

// --- START LOGIC COPY (German prompt assembly) ---
const userQuestion = $('Telegram Trigger').item.json.message.text;

const sources = items.map((item, index) => {
    const payload = item.json.payload || item.json;
    const text = payload.text_content || payload.content || "";
    const meta = payload.metadata || {};
    const citation = `[${meta.masechet || "Source"} ${meta.daf || index + 1}]`;

    return `${citation}\n${text}`;
}).join("\n\n--------------------\n\n");

const systemPrompt = `Du bist ein Talmud-Studienassistent.
Du musst NUR anhand der untenstehenden Quellen antworten.
Wenn die Antwort nicht explizit im bereitgestellten Text steht, antworte GENAU: "Keine explizite Quelle gefunden".
Verwende kein externes Wissen.
Zitiere deine Quellen im Format [Masechet Daf].

Quellen:
${sources}
`;

const result = {
    system_prompt: systemPrompt,
    user_prompt: `Frage: ${userQuestion}`
};
// --- END LOGIC COPY ---

console.log("=== Generated German System Prompt ===");
console.log(result.system_prompt);
console.log("\n=== Generated German User Prompt ===");
console.log(result.user_prompt);

// Validation checks
let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        console.log(`PASS: ${message}`);
        passed++;
    } else {
        console.log(`FAIL: ${message}`);
        failed++;
    }
}

assert(result.system_prompt.includes("Du bist"), "System prompt is in German");
assert(result.system_prompt.includes("Quellen:"), "Contains German 'Quellen' header");
assert(result.system_prompt.includes("[Berakhot 20a]"), "Contains source citation");
assert(result.system_prompt.includes("Keine explizite Quelle gefunden"), "Contains German fallback message");
assert(result.user_prompt.includes("Frage:"), "User prompt uses German 'Frage' prefix");
assert(result.user_prompt.includes("Schma"), "Preserves German user question");

console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
