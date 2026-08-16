import fs from 'fs'

const url = 'https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?ts=1786620958689'
const headers = {
  'Accept': 'application/json, text/plain, */*',
  'Origin': 'https://bdgwin888.com',
  'Referer': 'https://bdgwin888.com/',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36'
}

try {
  const res = await fetch(url, { headers })
  const text = await res.text()
  try {
    const json = JSON.parse(text)
    fs.writeFileSync('history.json', JSON.stringify(json, null, 2))
    console.log('Saved history.json (parsed JSON)')
  } catch (err) {
    fs.writeFileSync('history.json', text)
    console.log('Saved history.json (raw text)')
  }
} catch (err) {
  console.error('Fetch failed:', err.message)
  process.exit(1)
}
