const https = require('http');

const data = JSON.stringify({
  vectors: {
    size: 1536,
    distance: "Cosine"
  }
});

const options = {
  hostname: 'localhost',
  port: 6333,
  path: '/collections/talmud_vectors',
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': data.length
  }
};

const req = https.request(options, (res) => {
  console.log(`Status Code: ${res.statusCode}`);
  res.on('data', (d) => {
    process.stdout.write(d);
  });
});

req.on('error', (error) => {
  console.error(error);
});

req.write(data);
req.end();
