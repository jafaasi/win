import React, { useState } from 'react';

function toBigSmall(n) {
  return Number(n) >= 5 ? 'Big' : 'Small';
}

export default function HistoryLog({
  history = [],
  roundLogs = [],
  beadPlate = [],
  onReset,
  streakInfo,
  hotCold
}) {
  const [activeTab, setActiveTab] = useState('winloss'); // 'winloss' | 'bead' | 'stats'

  const displayHistory = history.slice(-30);
  const bigCount = history.filter(n => Number(n) >= 5).length;
  const smallCount = history.filter(n => Number(n) < 5).length;
  const bigPct = history.length > 0 ? Math.round((bigCount / history.length) * 100) : 0;
  const smallPct = history.length > 0 ? Math.round((smallCount / history.length) * 100) : 0;

  const displayBeadPlate = beadPlate.slice(-24);

  // Win/Loss Metrics
  const totalVerified = roundLogs.length;
  const winsCount = roundLogs.filter(r => r.isWin).length;
  const lossesCount = roundLogs.filter(r => !r.isWin).length;
  const winRate = totalVerified > 0 ? ((winsCount / totalVerified) * 100).toFixed(1) : '100.0';

  return (
    <div className="card fade-up fade-up-delay-3">
      {/* Header */}
      <div className="history-header">
        <div className="card-label" style={{ margin: 0 }}>
          <span className="label-icon">◫</span> CASINO LOGS & ROADMAP
        </div>
        {(history.length > 0 || roundLogs.length > 0) && (
          <button className="btn-ghost" onClick={onReset}>
            Clear
          </button>
        )}
      </div>

      {/* Win/Loss Scoreboard */}
      <div className="winloss-scoreboard">
        <div className="wl-score-item">
          <span className="wl-score-lbl">VERIFIED</span>
          <span className="wl-score-val">{totalVerified}</span>
        </div>
        <div className="wl-score-item win">
          <span className="wl-score-lbl">WINS</span>
          <span className="wl-score-val win">{winsCount}</span>
        </div>
        <div className="wl-score-item loss">
          <span className="wl-score-lbl">LOSSES</span>
          <span className="wl-score-val loss">{lossesCount}</span>
        </div>
        <div className="wl-score-item rate">
          <span className="wl-score-lbl">WIN RATE</span>
          <span className="wl-score-val rate">{winRate}%</span>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="history-tabs">
        <button
          className={`hist-tab-btn ${activeTab === 'winloss' ? 'active' : ''}`}
          onClick={() => setActiveTab('winloss')}
        >
          🎯 Win / Loss Log ({totalVerified})
        </button>
        <button
          className={`hist-tab-btn ${activeTab === 'bead' ? 'active' : ''}`}
          onClick={() => setActiveTab('bead')}
        >
          ◫ Bead Roadmap ({history.length})
        </button>
        <button
          className={`hist-tab-btn ${activeTab === 'stats' ? 'active' : ''}`}
          onClick={() => setActiveTab('stats')}
        >
          📊 Stats & Hot/Cold
        </button>
      </div>

      {/* Tab 1: Win / Loss History */}
      {activeTab === 'winloss' && (
        <div className="winloss-tab-content">
          {roundLogs.length === 0 ? (
            <div className="empty-tab-msg">
              No verified rounds yet. As each 30s draw arrives, win/loss verification logs will appear here live!
            </div>
          ) : (
            <div className="round-logs-list">
              {roundLogs.map((log) => (
                <div key={log.id} className={`round-log-card ${log.isWin ? 'win' : 'loss'}`}>
                  <div className="log-top-row">
                    <span className="log-issue">{log.issue}</span>
                    <span className={`log-badge ${log.isWin ? 'win' : 'loss'}`}>
                      {log.isWin ? '✓ WIN' : '✗ LOSS'}
                    </span>
                  </div>

                  <div className="log-detail-row">
                    <div className="log-target">
                      <span className="log-sublbl">Target:</span>
                      <span className={`log-val ${log.targetBS === 'Big' ? 'big' : 'small'}`}>
                        {log.targetBS} {log.targetNum !== undefined ? `[${log.targetNum}]` : ''}
                      </span>
                    </div>

                    <div className="log-arrow">→</div>

                    <div className="log-actual">
                      <span className="log-sublbl">Actual:</span>
                      <span className={`log-val ${log.actualBS === 'Big' ? 'big' : 'small'}`}>
                        {log.actualNum} ({log.actualBS})
                      </span>
                    </div>

                    <div className="log-level">
                      <span className={`level-pill level-${log.level || 1}`}>
                        L{log.level || 1}
                      </span>
                    </div>
                  </div>

                  <div className="log-footer-row">
                    <span className="log-pattern">{log.pattern}</span>
                    <span className="log-time">{log.time}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Bead Roadmap Matrix */}
      {activeTab === 'bead' && (
        <div className="bead-tab-content">
          {displayBeadPlate.length === 0 ? (
            <div className="empty-tab-msg">No rounds recorded for Bead Plate matrix yet.</div>
          ) : (
            <div className="bead-plate-section">
              <div className="bead-plate-title">MACAU BEAD PLATE MATRIX (LAST 24)</div>
              <div className="bead-plate-grid">
                {displayBeadPlate.map((item, idx) => {
                  const isBig = item.type === 'Big';
                  return (
                    <div
                      key={idx}
                      className={`bead-item ${item.color} ${isBig ? 'type-big' : 'type-small'}`}
                      title={`Round: ${item.number} (${item.type} · ${item.colorName})`}
                    >
                      <span className="bead-num">{item.number}</span>
                      <span className="bead-sub">{isBig ? 'B' : 'S'}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Sequence & Stats */}
      {activeTab === 'stats' && (
        <div className="stats-tab-content">
          {displayHistory.length === 0 ? (
            <div className="empty-tab-msg">No outcome data recorded yet.</div>
          ) : (
            <>
              <div className="history-pills">
                {displayHistory.map((num, index) => {
                  const isBig = Number(num) >= 5;
                  return (
                    <div
                      key={`${history.length}-${index}`}
                      className={`history-pill ${isBig ? 'big' : 'small'}`}
                      style={{ animationDelay: `${index * 0.02}s` }}
                      title={`Round ${history.length - displayHistory.length + index + 1}: ${num} (${toBigSmall(num)})`}
                    >
                      {num}
                    </div>
                  );
                })}
              </div>

              <div className="history-stats">
                <div className="stat-item">
                  <div className="stat-dot big" />
                  <span>Big:</span>
                  <span className="stat-value">{bigCount}</span>
                  <span>({bigPct}%)</span>
                </div>
                <div className="stat-item">
                  <div className="stat-dot small" />
                  <span>Small:</span>
                  <span className="stat-value">{smallCount}</span>
                  <span>({smallPct}%)</span>
                </div>
              </div>

              {hotCold?.hot?.length > 0 && hotCold?.cold?.length > 0 && (
                <div className="hot-cold-section">
                  <div className="hc-row">
                    <span className="hc-label hot">🔥 Hot Numbers</span>
                    <div className="hc-numbers">
                      {hotCold.hot.map(({ number, count }) => (
                        <span key={number} className={`hc-chip ${number >= 5 ? 'big' : 'small'}`}>
                          {number}<sup>{count}</sup>
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="hc-row">
                    <span className="hc-label cold">❄️ Cold Numbers</span>
                    <div className="hc-numbers">
                      {hotCold.cold.map(({ number, count }) => (
                        <span key={number} className={`hc-chip muted`}>
                          {number}<sup>{count}</sup>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {streakInfo && streakInfo.current >= 2 && (
                <div className="streak-bar">
                  <span className="streak-icon">
                    {streakInfo.currentOutcome === 'Big' ? '🔥' : '❄️'}
                  </span>
                  <span className="streak-text">
                    Active Dragon Streak: <span className="streak-count">{streakInfo.current}x {streakInfo.currentOutcome}</span>
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
