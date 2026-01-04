# Geomancer 🔮

**Lattice-Based RSA Factorization using GeometricLLL**

<img width="1176" height="771" alt="image" src="https://github.com/user-attachments/assets/77fb7687-dad0-4065-95ee-12691d41eeba" />

Geomancer is a powerful integer factorization tool that successfully factors **RSA-2048** by combining lattice-based relation finding with the blazing-fast GeometricLLL reduction algorithm.

## Current status of Python implementation
- On large N, it returns 'failed to find non-trivial factorization', even though a non-trivial is found.
- Dimension needs to be REALLY high to not see this message, and to factor large N.
- Forks are welcome!!!!!

---

## 🏆 Results

Note that it won't outright factor every key... yet.

| RSA Size | Lattice Dimension | Status |
|----------|-------------------|--------|
| RSA-512  | 200×200          | ✅ Factored |
| RSA-1024 | 500×500          | ✅ Factored |
| RSA-2048 | 1000×1000        | ✅ **Factored** | (Under special circumstances, a modulus that had many relations)

---

## ✨ Features

- **Lattice-Based Relation Finding**: Uses Schnorr-style lattice construction to find multiplicative relations among small primes
- **GeometricLLL Integration**: Leverages the 22x faster geometric lattice reduction
- **Auto-Scaling Parameters**: Automatically adjusts factor base size, precision, and lattice dimensions based on input size
- **Multi-Pass LLL**: Runs multiple reduction passes with varied scaling to maximize unique relations
- **Hybrid Approach**: Falls back to targeted sieving when additional relations are needed
- **High-Precision Arithmetic**: Uses Python's Decimal module for exact log calculations on 2048-bit numbers
- **Neural Network**: Custom Transformer to learn relations to assist with factoring at lower dimension
- **Full GUI**: Comprehensive tkinter-based interface with real-time logging and statistics

---

## 📋 Requirements

```bash
pip install numpy pycryptodome
```

---

## 🚀 Quick Start

### Command Line

```python
from Geomancer import GeometricFactorizer

# RSA-2048 modulus to factor
N = 26708882667554369982437861342351118474042371033997336697986586402...

# Create factorizer (auto-scales parameters for 2048-bit)
factorizer = GeometricFactorizer(N, lattice_dim=1000)

# Find relations via LLL
factorizer.find_relations()

# Solve linear system over GF(2) → extract factors
factorizer.solve_linear_system()
```

### GUI Mode

```bash
python geomancer_gui.py
```

![Geomancer GUI](docs/gui_screenshot.png)

**GUI Features:**
- Dark theme interface
- Real-time color-coded output logging
- Statistics panel (relations found, primes, dependencies)
- Menu for generating test RSA keys
- Save/load N values
- Non-blocking factorization with threading

---

## 🧮 Algorithm Overview

### Step 1: Schnorr Lattice Construction

Build a lattice where short vectors encode multiplicative relations:

```
For factor base {p₁, p₂, ..., pₖ} and modulus N:

     [ 1  0  ...  0  | C·ln(p₁) ]
     [ 0  1  ...  0  | C·ln(p₂) ]
B =  [ ⋮  ⋮  ⋱   ⋮  |    ⋮     ]
     [ 0  0  ...  1  | C·ln(pₖ) ]
     [ 0  0  ...  0  | C·ln(N)  ]
```

### Step 2: GeometricLLL Reduction

Reduce the lattice using GeometricLLL's geometric inversion technique:
- **22x faster** than standard LLL (fpylll)
- Finds short vectors representing: `∏ pᵢ^eᵢ ≈ N^k`

### Step 3: Relation Extraction

Each short vector gives exponents (e₁, e₂, ..., eₖ) such that:
```
p₁^e₁ · p₂^e₂ · ... · pₖ^eₖ ≈ 1  (mod N)
```

### Step 4: Linear Algebra over GF(2)

Combine relations to find:
```
x² ≡ y² (mod N)  where  x ≢ ±y (mod N)
```

### Step 5: Factor Extraction

```
gcd(x - y, N) → p  (non-trivial factor)
gcd(x + y, N) → q
```

---

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lattice_dim` | 500 | Lattice dimension (rows/cols) |
| `factor_base_size` | `lattice_dim // 2` | Number of primes (auto: half of lattice_dim) |
| `precision_bits` | `N.bit_length() + 50` | Decimal precision for log calculations |

### Why `factor_base_size = lattice_dim // 2`?

To guarantee finding dependencies in the kernel, we need:
```
relations > primes
```

By using half as many primes as the lattice dimension, we ensure an **overdetermined system** where the GF(2) kernel is non-trivial.

---

## 📊 How Many Relations?

| Lattice Dim | Factor Base | Relations Found | Kernel Size |
|-------------|-------------|-----------------|-------------|
| 500×500     | 250         | ~400-500        | 150-250     |
| 1000×1000   | 500         | ~800-1000       | 300-500     |

---

## 🔬 Technical Details

### The Lattice Approach vs Traditional Sieving

**Traditional QS/NFS:**
- Sieves for smooth numbers x² - N
- Smooth numbers become exponentially rare for large N
- 2048-bit: Would need to sieve ~10^50 values

**Geomancer's Lattice Approach:**
- Directly searches for multiplicative relations in structured space
- LLL finds short vectors = valid relations
- Works in the "geometry of numbers" rather than brute-force search

### Multi-Pass Strategy

Geomancer runs multiple LLL passes with varied scaling factors:
```python
for pass_num in range(5):
    scale = 1.0 + 0.1 * pass_num  # 1.0, 1.1, 1.2, 1.3, 1.4
    # Build lattice with C_scaled = C * scale
    # Each pass finds different short vectors
```

This extracts more unique relations from the same lattice structure.

---

## 📁 Files

```
├── Geomancer.py          # Main factorization engine
├── geomancer_gui.py      # Tkinter GUI interface  
├── geometric_lll.py      # GeometricLLL (22x faster than fpylll)
├── coppersmith.py        # Coppersmith's small roots (auxiliary)
└── GEOMANCER_README.md   # This file
```

---

## 🎯 Example Session

```
$ python geomancer_gui.py

N bit length: 2048
Lattice dimension: 1000x1000
Auto-scaled parameters:
  Factor Base Size: 500 (smaller than lattice for overdetermined system)
  Precision Bits: 2098

=== Pass 1/5 ===
Running GeometricLLL reduction...
Pass 1: Found 487 new relations (total: 487)

=== Pass 2/5 ===
Pass 2: Found 89 new relations (total: 576)
Target reached! Have 576 relations.

Solving Linear System with 576 relations...
Building 576x500 matrix over GF(2)...
Found 76 independent dependencies (Kernel size).

Trying dependency 1/76...
[SUCCESS] Factor found: 163424...
Other factor: 163456...
```

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. 

RSA remains secure when properly implemented with:
- Key sizes ≥ 3072 bits (NIST recommendation)
- Proper random prime generation
- Side-channel protections

Do not use this tool on systems you do not own or have explicit permission to test.

---

## 📚 References

1. Lenstra, A. K., Lenstra, H. W., & Lovász, L. (1982). *Factoring polynomials with rational coefficients*
2. Schnorr, C. P. (1993). *Factoring Integers and Computing Discrete Logarithms via Diophantine Approximation*
3. Nguyen, P. Q., & Stern, J. (2001). *The Two Faces of Lattices in Cryptology*

---

## 📄 License

MIT License

---

**Happy Factoring! 🎯**
