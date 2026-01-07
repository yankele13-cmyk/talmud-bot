// Test Script for Ingestion Logic
// Usage: node tests/test_ingestion_logic.js

const https = require('https');

// Mock n8n 'items' structure
function createMockItems(jsonPayload) {
    return [{ json: jsonPayload }];
}

// 1. Fetch from Sefaria
const url = "https://www.sefaria.org/api/texts/Berakhot.2a?context=0&commentary=0";

console.log(`Fetching ${url}...`);

https.get(url, (res) => {
    let data = '';
    res.on('data', (chunk) => data += chunk);
    res.on('end', () => {
        const sefariaJson = JSON.parse(data);
        console.log("Fetched Data. Processing...");

        // ==========================================
        // STEP 1: PARSING LOGIC (Copy of parsing_logic.js)
        // ==========================================
        let items = createMockItems(sefariaJson);
        
        // --- START PARSE CODE ---
        const responseData = items[0].json;
        const masechet = responseData.indexTitle;
        const heMasechet = responseData.heIndexTitle;
        
        // Fix for Berakhot.2a specific response structure
        // Sefaria returns 'text' and 'he' as arrays of strings for a single Daf request
        const textArray = responseData.text; 
        const heTextArray = responseData.he; 

        const results = [];
        for (const item of items) {
            const data = item.json;
            // if (!data.he || data.he.length === 0) continue; // Commented out for robust testing if empty

            let rawText = "";
            // Handle both Array of Strings (Daf request) and Array of Arrays (Chapter request)
            if (Array.isArray(data.he)) {
                // If it's 2D array (Chapters), flat it. If 1D (Daf), just join.
                // Sefaria "text" for a single Daf is usually ["Line 1", "Line 2"...] -> 1D Array
                // Sefaria "text" for Chapter is [["Line 1"], ["Line 2"]] -> 2D Array
                // Simple flatten usually works for both.
                rawText = data.he.flat(999).filter(t => t).join("\n");
            } else {
                rawText = data.he || "";
            }

            const cleanText = rawText.replace(/<[^>]*>?/gm, '');

            const ref = data.ref || "Unknown";
            const dafMatch = ref.match(/\d+[ab]/);
            const daf = dafMatch ? dafMatch[0] : ref;

            results.push({
                json: {
                    masechet: masechet || "Berakhot",
                    heMasechet: heMasechet || "ברכות",
                    daf: daf,
                    ref: ref,
                    type: "Gemara",
                    language: "he",
                    content: cleanText
                }
            });
        }
        items = results;
        // --- END PARSE CODE ---

        console.log("Parsed Item 1:", JSON.stringify(items[0].json, null, 2));

        // ==========================================
        // STEP 2: TEMPLATING LOGIC (Copy of templating_logic.js)
        // ==========================================
        
        // --- START TEMPLATE CODE ---
        for (const item of items) {
            const data = item.json;
            const template = `
מסכת: ${data.heMasechet}
דף: ${data.daf}
סוג: גמרא
שפה: עברית

====================
טקסט
====================
${data.content}

====================
סיום
====================
`;
            item.json.formatted_text = template.trim();
            item.json.metadata = {
                masechet: data.masechet,
                daf: data.daf,
                ref: data.ref,
                type: "Gemara",
                language: "he",
                url: `https://www.sefaria.org/${data.ref.replace(/ /g, '_')}` 
            };
        }
        // --- END TEMPLATE CODE ---

        console.log("\nFormatted Text Preview:\n-----------------------------------");
        console.log(items[0].json.formatted_text);
        console.log("-----------------------------------");
        
        // ==========================================
        // STEP 3: UUID LOGIC (Copy of uuid_helper.js)
        // ==========================================
        
        function stringToUuid(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                const char = str.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash; 
            }
            const hex = Math.abs(hash).toString(16).padStart(32, '0');
            return `${hex.substr(0, 8)}-${hex.substr(8, 4)}-${hex.substr(12, 4)}-${hex.substr(16, 4)}-${hex.substr(20, 12)}`;
        }

        for (const item of items) {
             const ref = item.json.ref || "unknown";
             item.json.uuid = stringToUuid(ref);
        }
        
        console.log("\nFinal UUID:", items[0].json.uuid);

    });
}).on('error', (e) => {
  console.error(e);
});
