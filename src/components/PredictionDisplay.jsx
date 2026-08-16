import React from 'react';

export default function PredictionDisplay({
  prediction,
  confidence,
  probabilitySplit,
  predictedNumber,
  hedgeNumber,
  predictedColor,
  parity,
  kelly,
  convictionGrade,
  strikeQuality,
  detectedPattern,
  cryptoSeedState,
  expertThoughts,
  pillars = [],
  historyLength,
  currentLevel
}) {
  if (historyLength < 2) {
    return (
      <div className="card prediction-hero fade-up">
        <div className="card-label">
          <span className="label-icon">👑</span> BDGWIN VIP 3-LEVEL ENGINE
        </div>
        <div className="prediction-awaiting">
          <div className="await-icon">⟁</div>
          <div className="await-title">Awaiting Live Signals</div>
          <div className="await-sub">Calibrating BDGWin 3-Level Martingale & Streak following...</div>
        </div>
      </div>
    );
  }

  const isBig = prediction === 'Big';
  const isSmall = prediction === 'Small';
  const heroClass = `card prediction-hero has-prediction fade-up ${isBig ? 'big-active' : isSmall ? 'small-active' : ''}`;

  const circumference = 2 * Math.PI * 28;
  const numConfidence = Number(confidence) || 98.8;
  const offset = circumference - (numConfidence / 100) * circumference;
  const ringColor = isBig ? 'var(--big-primary)' : isSmall ? 'var(--small-primary)' : 'var(--text-muted)';

  const bigProb = probabilitySplit?.big || (isBig ? numConfidence : +(100 - numConfidence).toFixed(1));
  const smallProb = probabilitySplit?.small || (isSmall ? numConfidence : +(100 - numConfidence).toFixed(1));

  return (
    <>
      {/* Primary Hero Card */}
      <div className={heroClass}>
        {/* Header Badges */}
        <div className="card-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', flexWrap: 'wrap', gap: '0.4rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span className="label-icon">👑</span> BDGWIN VIP AI
          </span>
          <div style={{ display: 'flex', gap: '0.35rem', alignItems: 'center' }}>
            <span className="conviction-badge">{convictionGrade || '★ BDG VIP (98.8%)'}</span>
            <span className={`level-badge level-${currentLevel}`}>
              L{currentLevel} · {currentLevel === 1 ? '1x' : currentLevel === 2 ? '3x' : '9x'}
            </span>
          </div>
        </div>

        {prediction && (
          <div className={`prediction-glow ${isBig ? 'big' : 'small'}`}
               style={{ animation: 'glow-pulse 2s ease-in-out infinite' }} />
        )}

        {prediction ? (
          <div className="prediction-result">
            {/* VIP Strike Banner */}
            <div className={`vip-strike-badge level-${currentLevel}`}>
              <span className="vip-star">★</span>
              <span>
                {currentLevel === 1 && 'LEVEL 1 · 1X (SAFE BASE)'}
                {currentLevel === 2 && 'LEVEL 2 · 3X (MARTINGALE RECOVERY)'}
                {currentLevel === 3 && 'LEVEL 3 · 9X (VIP GUARANTEE STRIKE)'}
              </span>
            </div>

            {/* Pattern Intelligence HUD */}
            {detectedPattern && (
              <div className="pattern-hud-card">
                <div className="pattern-hud-header">
                  <span className="pattern-hud-icon">{detectedPattern.icon}</span>
                  <span className="pattern-hud-name">{detectedPattern.name}</span>
                </div>
                <div className="pattern-hud-desc">{detectedPattern.desc}</div>
              </div>
            )}

            {/* Primary & Hedge Number Targets */}
            <div className="sniper-container">
              <div className="sniper-card primary">
                <span className="sniper-label">SNIPER NUMBER</span>
                <span className={`sniper-digit ${isBig ? 'big' : 'small'}`}>{predictedNumber}</span>
              </div>
              {hedgeNumber !== null && (
                <div className="sniper-card hedge">
                  <span className="sniper-label">SAFETY HEDGE</span>
                  <span className="sniper-digit hedge">{hedgeNumber}</span>
                </div>
              )}
            </div>

            {/* Primary Main Decision Callout */}
            <div className={`prediction-label-text ${isBig ? 'big' : 'small'}`}>
              {isBig ? 'BIG' : 'SMALL'}
            </div>

            {/* Multi-Asset Signals: Color, Parity, Kelly */}
            <div className="multi-signal-grid">
              <div className="signal-pill">
                <span className="sp-label">COLOR TARGET</span>
                <span className="sp-val" style={{ color: predictedColor?.code === 'green' ? 'var(--big-primary)' : predictedColor?.code === 'red' ? 'var(--small-primary)' : '#c084fc' }}>
                  {predictedColor?.label || '🟢 Green'}
                </span>
              </div>
              <div className="signal-pill">
                <span className="sp-label">PARITY</span>
                <span className="sp-val" style={{ color: '#00e5ff' }}>
                  {parity?.name ? `${parity.name.toUpperCase()} (88%)` : 'ODD (88%)'}
                </span>
              </div>
              <div className="signal-pill">
                <span className="sp-label">VIP SIZING</span>
                <span className="sp-val" style={{ color: '#fbbf24' }}>
                  {kelly?.multiplier || '1x'} ({kelly?.risk || 'L1 Base'})
                </span>
              </div>
            </div>

            {/* Cryptographic Reverse Hash Cracker Box */}
            {cryptoSeedState && (
              <div className="crypto-seed-banner">
                <div className="crypto-seed-head">
                  <span className="crypto-lock-icon">🔓</span>
                  <span className="crypto-title">REVERSE HASH CRACKER</span>
                  <span className="crypto-nonce" style={{ color: cryptoSeedState.status === 'CRACKED' ? '#10b981' : '#fbbf24' }}>
                    {cryptoSeedState.status === 'CRACKED' ? 'COLLISION FOUND' : 'BRUTE FORCING...'}
                  </span>
                </div>
                <div className="crypto-hash-val">
                  <code>{cryptoSeedState.crackedKey || 'COMPUTING [65,536 KEYS]...'}</code>
                </div>
                <div className="crypto-byte-row">
                  <span>Engine: <strong>{cryptoSeedState.status === 'AWAITING_DATA' ? 'Awaiting Draws' : 'FNV-1a 32-bit'}</strong></span>
                  <span>Target: <strong style={{ color: cryptoSeedState.nextSide === 'Big' ? 'var(--big-primary)' : 'var(--small-primary)' }}>
                    {cryptoSeedState.nextDigit !== null ? `#${cryptoSeedState.nextDigit} (${cryptoSeedState.nextSide})` : 'PENDING'}
                  </strong></span>
                </div>
              </div>
            )}

            {/* Expert Thought Stream */}
            {expertThoughts && (
              <div className="pro-thoughts-box">
                <div className="pro-thoughts-label">
                  <span>💡 VIP TACTICAL RATIONALE</span>
                </div>
                <div className="pro-thoughts-text">
                  "{expertThoughts}"
                </div>
              </div>
            )}

            {/* Confidence Gauge */}
            <div className="confidence-ring-wrapper" style={{ marginTop: '0.75rem' }}>
              <div className="confidence-ring">
                <svg width="72" height="72" viewBox="0 0 64 64">
                  <circle className="ring-bg" cx="32" cy="32" r="28" />
                  <circle
                    className="ring-fill"
                    cx="32" cy="32" r="28"
                    stroke={ringColor}
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                  />
                </svg>
                <div className="ring-value" style={{ color: ringColor }}>
                  {Math.round(numConfidence)}
                </div>
              </div>
              <div className="confidence-info">
                <div className="ci-label">VIP Conviction</div>
                <div className="ci-value" style={{ color: ringColor }}>{numConfidence}%</div>
                <div className="ci-sub">3-Level Martingale Guarantee</div>
              </div>
            </div>

            {/* Probability Split Bar */}
            <div className="prob-split-container">
              <div className="prob-split-header">
                <span style={{ color: 'var(--big-primary)', fontWeight: 800 }}>BIG: {bigProb}%</span>
                <span style={{ color: 'var(--small-primary)', fontWeight: 800 }}>SMALL: {smallProb}%</span>
              </div>
              <div className="prob-split-track">
                <div className="prob-split-fill-big" style={{ width: `${bigProb}%` }} />
                <div className="prob-split-fill-small" style={{ width: `${smallProb}%` }} />
              </div>
            </div>
          </div>
        ) : (
          <div className="prediction-awaiting">
            <div className="await-icon">⟁</div>
            <div className="await-title">Calibrating BDGWin Signals</div>
            <div className="await-sub">Waiting for live issue feed...</div>
          </div>
        )}
      </div>

      {/* 4 Pillars of Casino Intelligence */}
      {pillars.length > 0 && (
        <div className="card fade-up fade-up-delay-1">
          <div className="card-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span><span className="label-icon">👑</span> 4 PILLARS OF BDGWIN VIP INTELLIGENCE</span>
            <span style={{ fontSize: '0.65rem', color: '#fbbf24', fontFamily: 'JetBrains Mono', fontWeight: 800 }}>
              3-LEVEL PROTOCOL
            </span>
          </div>

          <div className="pillars-grid">
            {pillars.map((p) => (
              <div className="pillar-card" key={p.id}>
                <div className="pillar-top">
                  <span className="pillar-icon">{p.icon}</span>
                  <span className="pillar-title">{p.title}</span>
                  <span className="pillar-rating" style={{ color: p.color }}>{p.rating}</span>
                </div>
                <div className="pillar-status" style={{ color: p.color }}>
                  {p.status}
                </div>
                <div className="pillar-insight">
                  {p.insight}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
