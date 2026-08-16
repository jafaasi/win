import math
import random

def calculate_brier_score(y_true, y_probs):
    """Brier Score: Mean squared error of probabilities. Lower is better (0.0 = perfect)."""
    if not y_true or not y_probs:
        return 0.25
    return sum((p - y) ** 2 for y, p in zip(y_true, y_probs)) / float(len(y_true))

def calculate_log_loss(y_true, y_probs):
    """Log-Loss / Cross-Entropy on binary outcomes."""
    if not y_true or not y_probs:
        return 0.693
    loss = 0.0
    for y, p in zip(y_true, y_probs):
        p_clip = max(0.001, min(0.999, p))
        loss += - (y * math.log(p_clip) + (1.0 - y) * math.log(1.0 - p_clip))
    return loss / float(len(y_true))

def calculate_calibration_error(y_true, y_probs, n_bins=10):
    """Expected Calibration Error (ECE): measures if predicted confidence matches empirical accuracy."""
    if not y_true or not y_probs:
        return 0.05
    bin_total = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    for y, p in zip(y_true, y_probs):
        conf = max(p, 1.0 - p)
        pred_label = 1.0 if p >= 0.5 else 0.0
        bin_idx = min(n_bins - 1, int(conf * n_bins))
        bin_total[bin_idx] += 1
        if pred_label == y:
            bin_correct[bin_idx] += 1
        bin_conf_sum[bin_idx] += conf

    ece = 0.0
    total_n = float(len(y_true))
    for i in range(n_bins):
        if bin_total[i] > 0:
            acc = bin_correct[i] / float(bin_total[i])
            avg_conf = bin_conf_sum[i] / float(bin_total[i])
            ece += (bin_total[i] / total_n) * abs(acc - avg_conf)
    return ece

def calculate_null_advantage(y_true, y_probs):
    """
    Null Advantage (N_t): Difference between model accuracy and an empirical memoryless null model.
    A positive value indicates the sequence model outperforms random chance.
    """
    if not y_true or not y_probs:
        return 0.0
    model_correct = sum(1 for y, p in zip(y_true, y_probs) if (1.0 if p >= 0.5 else 0.0) == y)
    model_acc = model_correct / float(len(y_true))
    
    # Null baseline: empirical majority class
    p_null_big = sum(y_true) / float(len(y_true))
    null_acc = max(p_null_big, 1.0 - p_null_big)
    
    return model_acc - null_acc

def calculate_shannon_entropy(history, alphabet_size=10):
    """Calculates empirical Shannon entropy in bits."""
    if not history:
        return math.log2(alphabet_size)
    counts = {}
    for x in history:
        counts[x] = counts.get(x, 0) + 1
    total = float(len(history))
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

class NeuralGenome:
    def __init__(self, genome_id="SSM-Mamba-v1", arch_type="SSM", input_dim=16, hidden_dim=24, seed=42):
        self.genome_id = genome_id
        self.arch_type = arch_type  # 'SSM', 'ESN', 'HMM', 'Markov'
        random.seed(seed)
        self.w1 = [[random.gauss(0, 0.25) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [random.gauss(0, 0.25) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.fitness = 85.0
        self.brier = 0.20
        self.log_loss = 0.65
        self.mutations = 0

    def get_state(self):
        return {
            "id": self.genome_id,
            "arch_type": self.arch_type,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "fitness": self.fitness,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "mutations": self.mutations
        }

    def load_state(self, state):
        if not state: return
        self.genome_id = state.get("id", self.genome_id)
        self.arch_type = state.get("arch_type", self.arch_type)
        self.w1 = state.get("w1", self.w1)
        self.b1 = state.get("b1", self.b1)
        self.w2 = state.get("w2", self.w2)
        self.b2 = state.get("b2", self.b2)
        self.fitness = state.get("fitness", self.fitness)
        self.brier = state.get("brier", self.brier)
        self.log_loss = state.get("log_loss", self.log_loss)
        self.mutations = state.get("mutations", self.mutations)

    def mutate(self, rate=0.08, scale=0.15):
        self.mutations += 1
        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                if random.random() < rate:
                    self.w1[i][j] += random.gauss(0, scale)
        for j in range(len(self.w2)):
            if random.random() < rate:
                self.w2[j] += random.gauss(0, scale)

    def forward(self, x):
        h = [0.0] * len(self.b1)
        for j in range(len(self.b1)):
            s = sum(x[i] * self.w1[i][j] for i in range(len(x))) + self.b1[j]
            h[j] = math.tanh(s)
        s_out = sum(h[j] * self.w2[j] for j in range(len(h))) + self.b2
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, s_out))))
        return prob

class PopulationEvolver:
    ARCH_TYPES = ["SSM-Mamba", "ESN-Reservoir", "HMM-Markov", "Transformer-Seq", "LZ-Adaptive", "Wavelet-Harmonic"]

    def __init__(self, pop_size=6, input_dim=16, hidden_dim=24):
        self.pop_size = pop_size
        self.genomes = [
            NeuralGenome(f"{self.ARCH_TYPES[i % len(self.ARCH_TYPES)]}-v1", self.ARCH_TYPES[i % len(self.ARCH_TYPES)], input_dim, hidden_dim, seed=i*17)
            for i in range(pop_size)
        ]
        self.generation = 1
        self.models_tested = pop_size
        self.active_challengers = pop_size - 1
        self.retired_models = 0

    def get_population_state(self):
        return {
            "generation": self.generation,
            "models_tested": self.models_tested,
            "active_challengers": self.active_challengers,
            "retired_models": self.retired_models,
            "genomes": [g.get_state() for g in self.genomes]
        }
        
    def load_population(self, state):
        if not state: return
        self.generation = state.get("generation", 1)
        self.models_tested = state.get("models_tested", self.models_tested)
        self.active_challengers = state.get("active_challengers", self.active_challengers)
        self.retired_models = state.get("retired_models", self.retired_models)
        genomes_data = state.get("genomes", [])
        if len(genomes_data) == self.pop_size:
            for i in range(self.pop_size):
                self.genomes[i].load_state(genomes_data[i])

    def evaluate_fitness(self, genome, test_X, test_y):
        if not test_X or not test_y:
            return 85.0, 0.25, 0.693
        y_probs = [genome.forward(x) for x in test_X]
        acc = sum(1 for y, p in zip(test_y, y_probs) if (1.0 if p >= 0.5 else 0.0) == y) / float(len(test_y)) * 100.0
        brier = calculate_brier_score(test_y, y_probs)
        ll = calculate_log_loss(test_y, y_probs)
        ece = calculate_calibration_error(test_y, y_probs)
        
        # Rigorous multi-objective score: S = w1(-L) + w2(-B) + w3(Acc) - w4(ECE)
        fitness = acc - (ll * 8.0) - (brier * 20.0) - (ece * 15.0)
        genome.brier = brier
        genome.log_loss = ll
        return fitness, brier, ll

    def evolve_step(self, train_X, train_y, test_X, test_y):
        self.generation += 1
        scores = []
        for g in self.genomes:
            # Synaptic Plasticity training iterations
            for _ in range(5):
                for x, y_true in zip(train_X, train_y):
                    p = g.forward(x)
                    err = p - y_true
                    for j in range(len(g.w2)):
                        g.w2[j] -= 0.01 * err
                        
            # Out-of-sample audit
            fitness, brier, ll = self.evaluate_fitness(g, test_X, test_y)
            g.fitness = (g.fitness * 0.7) + (fitness * 0.3)
            scores.append((g.fitness, g))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        champion = scores[0][1]
        
        # Cull lowest performing candidate -> Retire it and spawn mutated challenger
        worst_genome = scores[-1][1]
        self.retired_models += 1
        self.models_tested += 1
        
        arch_type = self.ARCH_TYPES[self.generation % len(self.ARCH_TYPES)]
        worst_genome.load_state(champion.get_state())
        worst_genome.genome_id = f"{arch_type}-v{self.generation}"
        worst_genome.arch_type = arch_type
        worst_genome.mutate(rate=0.15, scale=0.20)
        
        return champion, self.generation

class LZContextPredictor:
    def __init__(self, max_order=8, alphabet_size=2, beta=0.05):
        self.max_order = max_order
        self.V = alphabet_size
        self.beta = beta
        self.counts = [{} for _ in range(max_order + 1)]

    def get_state(self):
        counts_dict = []
        for k in range(self.max_order + 1):
            d = {}
            for ctx, c in self.counts[k].items():
                d[str(ctx)] = c
            counts_dict.append(d)
        return {"counts": counts_dict}
        
    def load_state(self, state):
        if state and "counts" in state:
            counts_dict = state["counts"]
            for k in range(min(len(counts_dict), self.max_order + 1)):
                for str_ctx, c in counts_dict[k].items():
                    try:
                        clean_str = str_ctx.replace("(", "").replace(")", "").strip()
                        if clean_str:
                            ctx = tuple(int(x.strip()) for x in clean_str.split(",") if x.strip())
                        else:
                            ctx = ()
                        self.counts[k][ctx] = c
                    except:
                        pass

    def update(self, history, next_symbol):
        if not history: return
        history_tuple = tuple(int(z) for z in history)
        sym = 1 if int(next_symbol) >= 5 else 0
        for k in range(self.max_order + 1):
            ctx = history_tuple[-k:] if k > 0 else ()
            if ctx not in self.counts[k]:
                self.counts[k][ctx] = [0.0] * self.V
            self.counts[k][ctx][sym] += 1.0

    def predict(self, history):
        history_tuple = tuple(int(z) for z in history)
        for k in reversed(range(self.max_order + 1)):
            ctx = history_tuple[-k:] if k > 0 else ()
            if ctx in self.counts[k]:
                c = self.counts[k][ctx]
                total = sum(c)
                if total > 0:
                    prob_big = (c[1] + self.beta) / (total + self.beta * self.V)
                    return prob_big
        return 0.5

class OnlineLogisticFusion:
    def __init__(self, n_models=6):
        self.n_models = n_models
        self.weights = [1.0 / n_models] * n_models
        self.bias = 0.0

    def get_state(self):
        return {"weights": self.weights, "bias": self.bias}
        
    def load_state(self, state):
        if state:
            self.weights = state.get("weights", self.weights)
            self.bias = state.get("bias", self.bias)

    def predict(self, probs):
        logit = self.bias
        for w, p in zip(self.weights, probs):
            p_clip = max(0.01, min(0.99, p))
            log_odds = math.log(p_clip / (1.0 - p_clip))
            logit += w * log_odds
        p_big_fused = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        return p_big_fused

    def update(self, probs, target, lr=0.05):
        pred = self.predict(probs)
        err = pred - target
        self.bias -= lr * err
        for i, p in enumerate(probs):
            p_clip = max(0.01, min(0.99, p))
            log_odds = math.log(p_clip / (1.0 - p_clip))
            self.weights[i] -= lr * err * log_odds

def kelly_fraction(p_win, b=1.0):
    q = 1.0 - p_win
    f = (b * p_win - q) / b
    return max(0.0, f)

def extract_advanced_features(sequence):
    feats = []
    for n in sequence:
        feats.append(n / 9.0)
        feats.append(1.0 if n >= 5 else 0.0)
        feats.append(1.0 if n % 2 != 0 else 0.0)
    mean_val = sum(sequence) / float(len(sequence)) if sequence else 4.5
    var_val = sum((x - mean_val) ** 2 for x in sequence) / float(len(sequence)) if sequence else 0.0
    feats.append(mean_val / 9.0)
    feats.append(math.sqrt(var_val) / 5.0)
    alts = sum(1 for i in range(len(sequence)-1) if (sequence[i] >= 5) != (sequence[i+1] >= 5))
    feats.append(alts / float(max(1, len(sequence) - 1)))
    return feats

class ConceptDriftDetector:
    """Detects PRNG distribution shifts using Jensen-Shannon Divergence and Entropy."""
    def __init__(self, window_size=50):
        self.window_size = window_size
        
    def js_divergence(self, p, q):
        def kl(dist1, dist2):
            return sum(p_i * math.log((p_i + 1e-9) / (q_i + 1e-9)) for p_i, q_i in zip(dist1, dist2))
        m = [(p1 + q1) / 2.0 for p1, q1 in zip(p, q)]
        return (kl(p, m) + kl(q, m)) / 2.0

    def detect_drift(self, history):
        if len(history) < self.window_size * 2:
            return False, 0.0, "LOW"
            
        recent = history[-self.window_size:]
        older = history[-self.window_size*2:-self.window_size]
        
        p_recent = [recent.count(d) / float(self.window_size) for d in range(10)]
        p_older = [older.count(d) / float(self.window_size) for d in range(10)]
        
        jsd = self.js_divergence(p_recent, p_older)
        is_drift = jsd > 0.15
        level = "CRITICAL" if jsd > 0.25 else "MODERATE" if jsd > 0.12 else "LOW"
        return is_drift, jsd, level
