import fs from 'fs'

const predictor = await import('./src/utils/predictor.js');
const { predictNextOutcome } = predictor;

function extractFromHistoryFile(obj) {
  const list = obj?.data?.list || obj?.list || obj;
  if (!Array.isArray(list)) return [];
  const ordered = list.slice().sort((a, b) => {
    const ai = String(a.issueNumber || '').replace(/[^0-9]/g, '') || '0';
    const bi = String(b.issueNumber || '').replace(/[^0-9]/g, '') || '0';
    try { return Number(BigInt(ai) - BigInt(bi)); } catch { return ai.localeCompare(bi); }
  });
  return ordered.map(e => {
    const n = e.number ?? e.value ?? e.result ?? null;
    return n == null ? null : parseInt(String(n).match(/\d+/)?.[0] ?? NaN, 10);
  }).filter(x => !Number.isNaN(x) && x >= 0 && x <= 9);
}

try {
  const raw = JSON.parse(fs.readFileSync('history.json', 'utf8'));
  const numbers = extractFromHistoryFile(raw);
  console.log('Extracted history:', numbers.join(', '));
  const res = predictNextOutcome(numbers, 1);
  console.log('Prediction:');
  console.log(JSON.stringify(res, null, 2));
} catch (err) {
  console.error('Failed to run prediction:', err);
  process.exit(1);
}
