import json
from backend.database import save_ai_brain_state, load_ai_brain_state
from backend.evolution import (
    ConceptDriftDetector, PopulationEvolver, OnlineLogisticFusion, 
    LZContextPredictor, extract_advanced_features
)

def run_evoseq_cycle(history, db):
    if len(history) < 50:
        return None
        
    print("Running EVOSEQ Daily Evolution Cycle...")
    # 1. Detect Concept Drift
    detector = ConceptDriftDetector(window_size=50)
    is_drift, jsd = detector.detect_drift(history)
    
    if is_drift:
        print(f"⚠️ CONCEPT DRIFT DETECTED! JS Divergence: {jsd:.3f}. PRNG Regime has shifted. Retiring old models.")
    
    evolver = PopulationEvolver(pop_size=6)
    fusion = OnlineLogisticFusion(n_models=6)
    lz_predictor = LZContextPredictor(max_order=6)
    
    # Load Registry (State)
    brain = load_ai_brain_state(db)
    if brain and brain.synaptic_weights and not is_drift:
        try:
            state = json.loads(brain.synaptic_weights)
            evolver.load_population(state.get("evolver"))
            fusion.load_state(state.get("fusion"))
            lz_predictor.load_state(state.get("lz"))
        except Exception as e:
            print("Failed to load registry:", e)
            
    # Time-decayed learning (walk-forward over recent window)
    window_size = 3
    train_depth = min(300, len(history) - window_size - 1)
    
    # Fast-forward LZ on deep history
    for i in range(1, len(history) - train_depth):
        lz_predictor.update(history[:i], history[i])
        
    X, y = [], []
    for i in range(len(history) - train_depth, len(history) - window_size):
        seq = history[i:i + window_size]
        target = 1.0 if history[i + window_size] >= 5 else 0.0
        X.append(extract_advanced_features(seq))
        y.append(target)
        
    split_idx = int(len(X) * 0.8)
    train_X, train_y = X[:split_idx], y[:split_idx]
    test_X, test_y = X[split_idx:], y[split_idx:]
    
    # Evolve Neural Population
    if len(train_X) > 0 and len(test_X) > 0:
        champion, gen_num = evolver.evolve_step(train_X, train_y, test_X, test_y)
    else:
        champion = evolver.genomes[0]
        gen_num = 1
    
    # Save the Champion and Registry
    registry_state = {
        "evolver": evolver.get_population_state(),
        "fusion": fusion.get_state(),
        "lz": lz_predictor.get_state(),
        "champion_id": champion.genome_id,
        "fitness": champion.fitness,
        "js_divergence": jsd
    }
    
    save_ai_brain_state(
        db=db,
        model_name="EVOSEQ_Registry",
        generation=gen_num,
        total_samples=len(history),
        weights_json=json.dumps(registry_state),
        win_rate=champion.fitness
    )
    
    print(f"🏆 Crowned Champion: {champion.genome_id} | Fitness: {champion.fitness:.1f}%")
    return registry_state
