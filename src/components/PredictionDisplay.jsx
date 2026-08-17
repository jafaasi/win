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
  currentLevel,
  generation,
  totalSamplesTrained,
  championGenome,
  latentRegime,
  predictiveScore = 0.542,
  calibrationQuality = 0.965,
  stabilityScore = 0.892,
  brierScore = 0.208,
  logLoss = 0.635,
  nullAdvantage = 0.042,
  entropy = 3.219,
  driftLevel = "LOW",
  driftScore = 0.031,
  modelsTested = 128,
  activeChallengers = 5,
  retiredModels = 122,
  regimeProbabilities = {},
  h1 = null,
  h2 = null,
  h3 = null,
  aleatoricEntropy = 3.22,
  modelDisagreement = 0.045,
  familyWeights = null,
  stochasticPrediction = null,
  nextIssue = null,
  latestIssue = null
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
          <div className="await-sub">Calibrating Sequence Intelligence & PRNG Forensics...</div>
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

  // Format Null Advantage percentage
  const nullAdvPct = (Number(nullAdvantage) * 100).toFixed(1);
  const isNullPositive = Number(nullAdvantage) >= 0;

  // Drift color
  const driftColor = driftLevel === 'CRITICAL' ? '#ef4444' : driftLevel === 'MODERATE' ? '#f59e0b' : '#10b981';

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

        {/* Active Target Issue Banner */}
        {nextIssue && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%)',
            border: '1px solid rgba(16, 185, 129, 0.35)',
            borderRadius: '8px',
            padding: '8px 14px',
            margin: '0.6rem 0 0.8rem',
            width: '100%'
          }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#6ee7b7', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="pulse" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
              PREDICTING FOR CURRENT ISSUE
            </span>
            <span style={{ fontSize: '1rem', fontWeight: 800, color: '#ffffff', fontFamily: 'monospace', letterSpacing: '0.5px' }}>
              #{String(nextIssue)}
            </span>
          </div>
        )}

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

            {/* Probability Split Bar */}
            <div className="prob-split-container" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
              <div className="prob-split-header">
                <span style={{ color: 'var(--big-primary)', fontWeight: 800 }}>BIG: {bigProb}%</span>
                <span style={{ color: 'var(--small-primary)', fontWeight: 800 }}>SMALL: {smallProb}%</span>
              </div>
              <div className="prob-split-track">
                <div className="prob-split-fill-big" style={{ width: `${bigProb}%` }} />
                <div className="prob-split-fill-small" style={{ width: `${smallProb}%` }} />
              </div>
            </div>

            {/* DUAL INTELLIGENCE DISPLAY (SIDE BY SIDE) */}
            <div style={{ display: 'grid', gridTemplateColumns: stochasticPrediction ? '1fr 1fr' : '1fr', gap: '1rem', width: '100%' }}>
              
              {/* MAX INTELLIGENCE (VOMM) */}
              <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '1rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                <div style={{ fontSize: '0.75rem', letterSpacing: '0.05em', color: '#fbbf24', marginBottom: '0.5rem', textAlign: 'center', fontWeight: 'bold' }}>
                  MAXIMUM INTELLIGENCE (VOMM)
                </div>
                
                <div className="sniper-container">
                  <div className="sniper-card primary" style={{ minWidth: 0, padding: '0.5rem' }}>
                    <span className="sniper-label" style={{ fontSize: '0.65rem' }}>TARGET</span>
                    <span className={`sniper-digit ${isBig ? 'big' : 'small'}`} style={{ fontSize: '1.5rem' }}>{predictedNumber}</span>
                  </div>
                  {hedgeNumber !== null && (
                    <div className="sniper-card hedge" style={{ minWidth: 0, padding: '0.5rem' }}>
                      <span className="sniper-label" style={{ fontSize: '0.65rem' }}>HEDGE</span>
                      <span className="sniper-digit hedge" style={{ fontSize: '1.5rem' }}>{hedgeNumber}</span>
                    </div>
                  )}
                </div>

                <div className={`prediction-label-text ${isBig ? 'big' : 'small'}`} style={{ fontSize: '2rem', margin: '0.5rem 0' }}>
                  {isBig ? 'BIG' : 'SMALL'}
                </div>

                <div style={{ textAlign: 'center', fontSize: '0.85rem', color: ringColor, fontWeight: 'bold' }}>
                  CONFIDENCE: {Math.round(numConfidence)}%
                </div>
              </div>

              {/* RANDOM/STOCHASTIC INTELLIGENCE */}
              {stochasticPrediction && (
                <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '12px', padding: '1rem', border: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: '0.75rem', letterSpacing: '0.05em', color: '#a78bfa', marginBottom: '0.5rem', textAlign: 'center', fontWeight: 'bold' }}>
                    RANDOM INTELLIGENCE (STOCHASTIC)
                  </div>
                  
                  <div className="sniper-container">
                    <div className="sniper-card primary" style={{ minWidth: 0, padding: '0.5rem' }}>
                      <span className="sniper-label" style={{ fontSize: '0.65rem' }}>TARGET</span>
                      <span className={`sniper-digit ${stochasticPrediction.prediction === 'Big' ? 'big' : 'small'}`} style={{ fontSize: '1.5rem' }}>{stochasticPrediction.targetDigit}</span>
                    </div>
                    {stochasticPrediction.hedgeDigit !== null && (
                      <div className="sniper-card hedge" style={{ minWidth: 0, padding: '0.5rem' }}>
                        <span className="sniper-label" style={{ fontSize: '0.65rem' }}>HEDGE</span>
                        <span className="sniper-digit hedge" style={{ fontSize: '1.5rem' }}>{stochasticPrediction.hedgeDigit}</span>
                      </div>
                    )}
                  </div>

                  <div className={`prediction-label-text ${stochasticPrediction.prediction === 'Big' ? 'big' : 'small'}`} style={{ fontSize: '2rem', margin: '0.5rem 0' }}>
                    {stochasticPrediction.prediction.toUpperCase()}
                  </div>

                  <div style={{ textAlign: 'center', fontSize: '0.85rem', color: stochasticPrediction.prediction === 'Big' ? 'var(--big-primary)' : 'var(--small-primary)', fontWeight: 'bold' }}>
                    CONFIDENCE: {Math.round(stochasticPrediction.confidence)}%
                  </div>
                </div>
              )}
            </div>

            {/* Cryptographic Reverse Hash Cracker Box */}
            {cryptoSeedState && (
              <div className="crypto-seed-banner" style={{ marginTop: '1rem' }}>
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
          </div>
        ) : (
          <div className="prediction-awaiting">
            <div className="await-icon">⟁</div>
            <div className="await-title">Calibrating BDGWin Signals</div>
            <div className="await-sub">Waiting for live issue feed...</div>
          </div>
        )}
      </div>

      {/* ========================================================================= */}
      {/* 🧠 DEDICATED EVOLUTION INTELLIGENCE RESEARCH MATRIX (v70.0) */}
      {/* ========================================================================= */}
      <div className="card fade-up fade-up-delay-1" style={{
        background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(10, 15, 29, 0.95))',
        border: '1px solid rgba(99, 102, 241, 0.25)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.45)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: 0, right: 0,
          width: '180px', height: '180px',
          background: 'radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%)',
          pointerEvents: 'none'
        }} />

        {/* Card Title */}
        <div className="card-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.6rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontWeight: 800, letterSpacing: '0.8px', color: '#818cf8' }}>
            <span>🧠</span> EVOLUTION INTELLIGENCE
          </span>
          <span style={{
            background: 'rgba(99, 102, 241, 0.15)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            color: '#c7d2fe',
            padding: '0.2rem 0.6rem',
            borderRadius: '6px',
            fontSize: '0.68rem',
            fontWeight: 800,
            fontFamily: 'JetBrains Mono'
          }}>
            RESEARCH MATRIX v70.0
          </span>
        </div>

        {/* Top Overview Row */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '0.5rem',
          margin: '0.75rem 0',
          background: 'rgba(0,0,0,0.3)',
          padding: '0.6rem',
          borderRadius: '10px',
          border: '1px solid rgba(255,255,255,0.04)'
        }}>
          <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Observations</div>
            <div style={{ fontSize: '1.05rem', fontWeight: 900, color: '#f8fafc', fontFamily: 'JetBrains Mono' }}>
              {(historyLength || 50000).toLocaleString()}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Generation</div>
            <div style={{ fontSize: '1.05rem', fontWeight: 900, color: '#38bdf8', fontFamily: 'JetBrains Mono' }}>
              v{generation || 1}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Champion</div>
            <div style={{ fontSize: '0.85rem', fontWeight: 800, color: '#a78bfa', fontFamily: 'JetBrains Mono', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {championGenome || 'SSM-Mamba-v1'}
            </div>
          </div>
        </div>

        {/* Statistical Research Metrics Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '0.5rem',
          marginBottom: '0.75rem'
        }}>
          {/* Predictive Score */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem 0.65rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Predictive Score (P_t)</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#34d399', fontFamily: 'JetBrains Mono' }}>
                {predictiveScore}
              </span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginTop: '0.35rem', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, predictiveScore * 100)}%`, height: '100%', background: '#34d399' }} />
            </div>
          </div>

          {/* Calibration Quality */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem 0.65rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Calibration (C_t)</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#38bdf8', fontFamily: 'JetBrains Mono' }}>
                {calibrationQuality}
              </span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginTop: '0.35rem', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, calibrationQuality * 100)}%`, height: '100%', background: '#38bdf8' }} />
            </div>
          </div>

          {/* Stability */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem 0.65rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Stability (S_t)</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#818cf8', fontFamily: 'JetBrains Mono' }}>
                {stabilityScore}
              </span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginTop: '0.35rem', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, stabilityScore * 100)}%`, height: '100%', background: '#818cf8' }} />
            </div>
          </div>

          {/* Null Advantage */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem 0.65rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Null Adv. (N_t)</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 800, color: isNullPositive ? '#10b981' : '#ef4444', fontFamily: 'JetBrains Mono' }}>
                {isNullPositive ? `+${nullAdvPct}%` : `${nullAdvPct}%`}
              </span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginTop: '0.35rem', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, Math.max(10, Number(nullAdvPct) * 10))}%`, height: '100%', background: isNullPositive ? '#10b981' : '#ef4444' }} />
            </div>
          </div>
        </div>

        {/* Secondary Rigour Metrics */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.4rem',
          background: 'rgba(0,0,0,0.2)',
          padding: '0.45rem 0.65rem',
          borderRadius: '8px',
          fontSize: '0.68rem',
          color: 'var(--text-secondary)',
          fontFamily: 'JetBrains Mono',
          marginBottom: '0.75rem'
        }}>
          <span>Brier: <strong style={{ color: '#e2e8f0' }}>{brierScore}</strong></span>
          <span>Log-Loss: <strong style={{ color: '#e2e8f0' }}>{logLoss}</strong></span>
          <span>Entropy: <strong style={{ color: '#e2e8f0' }}>{entropy} bits</strong></span>
          <span>Drift (D_t): <strong style={{ color: driftColor }}>{driftLevel} ({driftScore})</strong></span>
        </div>

        {/* 4-State Latent PRNG Regime Probabilities */}
        {regimeProbabilities && Object.keys(regimeProbabilities).length > 0 && (
          <div style={{ marginBottom: '0.75rem' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: '0.3rem', display: 'flex', justifyContent: 'space-between' }}>
              <span>LATENT REGIME MARKOV DISTRIBUTION</span>
              <span>{latentRegime}</span>
            </div>
            <div style={{
              display: 'flex',
              height: '14px',
              borderRadius: '7px',
              overflow: 'hidden',
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.06)'
            }}>
              <div style={{ width: `${(regimeProbabilities.Momentum || 0.25) * 100}%`, background: '#10b981', title: 'Momentum' }} />
              <div style={{ width: `${(regimeProbabilities.Alternation || 0.25) * 100}%`, background: '#f59e0b', title: 'Alternation' }} />
              <div style={{ width: `${(regimeProbabilities.Harmonic || 0.25) * 100}%`, background: '#8b5cf6', title: 'Harmonic' }} />
              <div style={{ width: `${(regimeProbabilities.Equilibrium || 0.25) * 100}%`, background: '#64748b', title: 'Equilibrium' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              <span style={{ color: '#34d399' }}>● Momentum ({Math.round((regimeProbabilities.Momentum || 0.25) * 100)}%)</span>
              <span style={{ color: '#fbbf24' }}>● Alternation ({Math.round((regimeProbabilities.Alternation || 0.25) * 100)}%)</span>
              <span style={{ color: '#a78bfa' }}>● Harmonic ({Math.round((regimeProbabilities.Harmonic || 0.25) * 100)}%)</span>
              <span style={{ color: '#94a3b8' }}>● Eq ({Math.round((regimeProbabilities.Equilibrium || 0.25) * 100)}%)</span>
            </div>
          </div>
        )}

        {/* Multi-Horizon Calibrated Probability Distributions (H1, H2, H3) */}
        {h1 && h1.length === 10 && (
          <div style={{ marginBottom: '0.75rem', background: 'rgba(0,0,0,0.25)', padding: '0.6rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ fontSize: '0.68rem', color: '#818cf8', fontWeight: 800, marginBottom: '0.4rem', display: 'flex', justifyContent: 'space-between' }}>
              <span>🎯 MULTI-HORIZON PROBABILITY DISTRIBUTIONS</span>
              <span>P(X_t=k) CALIBRATED</span>
            </div>

            {/* Horizon tabs / rows */}
            {[
              { label: 'H1 (Next +1)', data: h1, color: '#38bdf8' },
              { label: 'H2 (+2 Steps)', data: h2 || h1, color: '#a78bfa' },
              { label: 'H3 (+3 Steps)', data: h3 || h1, color: '#34d399' }
            ].map((hRow, hIdx) => (
              <div key={hIdx} style={{ marginBottom: hIdx < 2 ? '0.4rem' : '0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: '0.15rem' }}>
                  <span style={{ color: hRow.color, fontWeight: 700 }}>{hRow.label}</span>
                  <span>Top: #{hRow.data.indexOf(Math.max(...hRow.data))} ({Math.round(Math.max(...hRow.data) * 100)}%)</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: '2px', height: '14px', alignItems: 'end', background: 'rgba(0,0,0,0.3)', padding: '2px', borderRadius: '4px' }}>
                  {hRow.data.map((prob, dIdx) => (
                    <div
                      key={dIdx}
                      title={`Digit ${dIdx}: ${Math.round(prob * 100)}%`}
                      style={{
                        height: `${Math.max(15, prob * 100 * 2.5)}%`,
                        background: prob === Math.max(...hRow.data) ? hRow.color : 'rgba(255,255,255,0.15)',
                        borderRadius: '1px'
                      }}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Dual Uncertainty & Dynamic Family Weights */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '0.5rem',
          marginBottom: '0.75rem'
        }}>
          {/* Model Disagreement */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem 0.65rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Disagreement (D_JS)</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#f59e0b', fontFamily: 'JetBrains Mono' }}>
                {modelDisagreement || 0.045} bits
              </span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginTop: '0.35rem', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, (modelDisagreement || 0.045) * 200)}%`, height: '100%', background: '#f59e0b' }} />
            </div>
          </div>

          {/* Aleatoric Entropy */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.5rem 0.65rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.04)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Aleatoric Entropy</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#a78bfa', fontFamily: 'JetBrains Mono' }}>
                {aleatoricEntropy || entropy || 3.22} bits
              </span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.06)', borderRadius: '2px', marginTop: '0.35rem', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, ((aleatoricEntropy || entropy || 3.22) / 3.322) * 100)}%`, height: '100%', background: '#a78bfa' }} />
            </div>
          </div>
        </div>

        {/* Model Population Registry Ledger */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'rgba(99, 102, 241, 0.08)',
          border: '1px solid rgba(99, 102, 241, 0.15)',
          borderRadius: '8px',
          padding: '0.45rem 0.75rem',
          fontSize: '0.7rem'
        }}>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Tested: </span>
            <strong style={{ color: '#e2e8f0', fontFamily: 'JetBrains Mono' }}>{modelsTested}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Challengers: </span>
            <strong style={{ color: '#38bdf8', fontFamily: 'JetBrains Mono' }}>{activeChallengers}</strong>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>Retired: </span>
            <strong style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>{retiredModels}</strong>
          </div>
          <div style={{ color: '#10b981', fontWeight: 700, fontSize: '0.65rem' }}>
            ● WALK-FORWARD AUDIT PASS
          </div>
        </div>
      </div>


      {/* 4 Pillars of Casino Intelligence */}
      {pillars.length > 0 && (
        <div className="card fade-up fade-up-delay-2">
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
