"""Test kickmix import and basic API in WSL."""
import sys
sys.path.insert(0, "/mnt/d/kaggle/google-verifiable-quantum-circuit-compilation/data/kickmix")
import kickmix as km

print("kickmix imported OK")
print("version:", getattr(km, "__version__", "n/a"))
print("top-level:", [n for n in dir(km) if not n.startswith("_")][:50])
c = km.CircuitBuilder()
x = c.create_quantum_register(4, name="x")
c.x(x[0])
circ = c.finish_circuit()
print("circuit num_qubits:", circ.num_qubits, "num_ops:", len(circ))
stats = circ.operation_statistics()
print("num_toffoli_like:", stats.num_toffoli_like)
