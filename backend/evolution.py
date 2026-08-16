import math
import random

class NeuralGenome:
    def __init__(self, genome_id="Alpha", input_dim=16, hidden_dim=24, seed=42):
        self.genome_id = genome_id
        random.seed(seed)
        self.w1 = [[random.gauss(0, 0.25) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [random.gauss(0, 0.25) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.fitness = 85.0
        self.mutations = 0

    def get_state(self):
        return {
            "id": self.genome_id,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "fitness": self.fitness,
            "mutations": self.mutations
        }

    def load_state(self, state):
        if not state: return
        self.genome_id = state.get("id", self.genome_id)
        self.w1 = state.get("w1", self.w1)
        self.b1 = state.get("b1", self.b1)
        self.w2 = state.get("w2", self.w2)
        self.b2 = state.get("b2", self.b2)
        self.fitness = state.get("fitness", self.fitness)
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
    def __init__(self, pop_size=6, input_dim=16, hidden_dim=24):
        self.pop_size = pop_size
        self.genomes = [NeuralGenome(f"Gen-{i}", input_dim, hidden_dim, seed=i) for i in range(pop_size)]
        self.generation = 1

    def get_population_state(self):
        return {
            "generation": self.generation,
            "genomes": [g.get_state() for g in self.genomes]
        }
        
    def load_population(self, state):
        if not state: return
        self.generation = state.get("generation", 1)
        genomes_data = state.get("genomes", [])
        if len(genomes_data) == self.pop_size:
            for i in range(self.pop_size):
                self.genomes[i].load_state(genomes_data[i])

    def evaluate_fitness(self, genome, test_X, test_y):
        correct = 0
        loss = 0.0
        for x, y_true in zip(test_X, test_y):
            p_big = genome.forward(x)
            pred = 1.0 if p_big >= 0.5 else 0.0
            if pred == y_true: correct += 1
            # Log Loss
            p_clip = max(0.01, min(0.99, p_big))
            loss += - (y_true * math.log(p_clip) + (1 - y_true) * math.log(1 - p_clip))
            
        acc = (correct / max(1, len(test_y))) * 100.0
        avg_loss = loss / max(1, len(test_y))
        # Fitness combines accuracy with a penalty for high log-loss (uncertainty)
        return acc - (avg_loss * 5.0)

    def evolve_step(self, train_X, train_y, test_X, test_y):
        self.generation += 1
        scores = []
        for g in self.genomes:
            # 1. Very rough SGD step for training (Synaptic Plasticity)
            for _ in range(5): 
                for x, y_true in zip(train_X, train_y):
                    p = g.forward(x)
                    err = p - y_true
                    for j in range(len(g.w2)):
                        g.w2[j] -= 0.01 * err
                        
            # Evaluate out-of-sample fitness
            fitness = self.evaluate_fitness(g, test_X, test_y)
            # EMA fitness update
            g.fitness = (g.fitness * 0.7) + (fitness * 0.3)
            scores.append((g.fitness, g))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        champion = scores[0][1]
        
        # 2. Darwinian Mutation: Cull the lowest performing genome and replace with mutated champion
        worst_genome = scores[-1][1]
        worst_genome.load_state(champion.get_state())
        worst_genome.genome_id = f"Mutant-{self.generation % 1000}"
        worst_genome.mutate(rate=0.15, scale=0.20)
        
        return champion, self.generation

class LZContextPredictor:
    def __init__(self, max_order=8, alphabet_size=2, beta=0.05):
        self.max_order = max_order
        self.V = alphabet_size
        self.beta = beta
        self.counts = [{} for _ in range(max_order + 1)]

    def get_state(self):
        # Convert tuples to string keys for JSON serialization
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
                    # Parse tuple string like "(1, 0, 1)" back to tuple
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
            return sum(p * math.log((p + 1e-9) / (q + 1e-9)) for p, q in zip(dist1, dist2))
        m = [(p1 + q1) / 2.0 for p1, q1 in zip(p, q)]
        return (kl(p, m) + kl(q, m)) / 2.0

    def detect_drift(self, history):
        if len(history) < self.window_size * 2:
            return False, 0.0
            
        recent = history[-self.window_size:]
        older = history[-self.window_size*2:-self.window_size]
        
        # Calculate digit distributions
        p_recent = [recent.count(d) / float(self.window_size) for d in range(10)]
        p_older = [older.count(d) / float(self.window_size) for d in range(10)]
        
        jsd = self.js_divergence(p_recent, p_older)
        # Empirical threshold for significant drift
        is_drift = jsd > 0.15 
        return is_drift, jsd
