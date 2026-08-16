# EVOSEQ Research Experiments & Null Referee Protocols

EVOSEQ treats sequence predictability as a scientific hypothesis requiring empirical out-of-sample evidence rather than manufactured certainty.

---

## 🛡️ Surrogate Null Models

To prevent overfitting to spurious variance, candidate models are benchmarked against 4 controlled null baselines:

1. **IID Empirical-Distribution Null ($H_{0,A}$)**:
   - Independent and identically distributed draws from marginal class frequencies $P(X)$.
2. **Permutation Null ($H_{0,B}$)**:
   - Preserves exact marginal counts while destroying all temporal orderings.
3. **Block-Shuffle Null ($H_{0,C}$)**:
   - Preserves short-term dependency blocks while randomizing long-range autocorrelation.
4. **Low-Order Markov Null ($H_{0,D}$)**:
   - Synthetic sequences generated from fitted 1st-order transition matrix.

---

## 📊 Empirical Significance p-Value

Candidate improvement score $S_{\text{real}}$ must demonstrate statistical significance against $N$ surrogate null scores $S_{\text{null}}^{(i)}$:

$$p = \frac{1 + \sum_{i=1}^N \mathbf{1}_{\{S_{\text{null}}^{(i)} \ge S_{\text{real}}\}}}{N + 1}$$

A candidate is promoted only when $p \le 0.10$.
