// Vercel serverless function for /api/state
export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    const AWS_API_URL = 'http://3.7.65.149:8000/api/state';
    
    console.log('Proxying request to:', AWS_API_URL);
    console.log('Request method:', req.method);
    console.log('Request body:', req.body);
    
    const response = await fetch(AWS_API_URL, {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: req.method === 'POST' ? JSON.stringify(req.body) : undefined,
    });

    console.log('AWS response status:', response.status);
    
    const text = await response.text();
    console.log('AWS response length:', text.length);
    
    let data;
    try {
      data = JSON.parse(text);
    } catch (e) {
      console.error('Failed to parse JSON:', text.substring(0, 200));
      data = { error: 'Invalid JSON response', raw: text.substring(0, 500) };
    }
    
    res.status(response.status).json(data);
  } catch (error) {
    console.error('Proxy error:', error);
    res.status(500).json({ error: 'Failed to proxy request to AWS', message: error.message });
  }
}