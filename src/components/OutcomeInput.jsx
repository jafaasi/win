import React from 'react';

export default function OutcomeInput({ onAddOutcome }) {
  const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];

  return (
    <div className="card fade-up fade-up-delay-2">
      <div className="card-label">
        <span className="label-icon">◉</span> ENTER OUTCOME
      </div>
      <div className="numpad-grid">
        {numbers.map(n => {
          const isBig = n >= 5;
          return (
            <button
              key={n}
              className={`numpad-btn ${isBig ? 'big' : 'small'}`}
              onClick={() => onAddOutcome(n)}
            >
              <span className="numpad-digit">{n}</span>
              <span className="numpad-tag">{isBig ? 'B' : 'S'}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
