import numpy as np
from Crypto.Util.number import getPrime, isPrime, GCD
from geometric_lll import GeometricLLL
import math
import random
from functools import reduce
from decimal import Decimal, getcontext

class GeometricFactorizer:
    def __init__(self, N, factor_base_size=None, precision_bits=None, interval_size=None, lattice_dim=500):
        self.N = N
        self.lattice_dim = lattice_dim
        
        # Scale parameters based on bit length of N
        n_bits = N.bit_length()
        
        # Factor base size: Use lattice_dim - 1 (one row reserved for N)
        if factor_base_size is None:
            factor_base_size = lattice_dim - 1
        
        # Precision bits: should be enough to distinguish log values
        # Need at least n_bits to capture the full precision of N
        if precision_bits is None:
            precision_bits = max(100, n_bits + 50)
        
        # Interval size for supplementary sieve
        if interval_size is None:
            interval_size = max(100000, min(10000000, factor_base_size * 100))
        
        self.precision_bits = precision_bits
        self.interval_size = interval_size
        self.primes = self._generate_factor_base(factor_base_size)
        self.C = 1 << precision_bits  # Scaling factor for logarithms
        self.relations = []
        
        print(f"N bit length: {n_bits}")
        print(f"Lattice dimension: {lattice_dim}x{lattice_dim}")
        print(f"Auto-scaled parameters:")
        print(f"  Factor Base Size: {factor_base_size}")
        print(f"  Precision Bits: {precision_bits}")
        print(f"  Interval Size: {interval_size}")

    def _tonelli_shanks(self, n, p):
        """
        Finds x such that x^2 = n (mod p).
        """
        if pow(n, (p - 1) // 2, p) != 1:
            return []
        
        if p % 4 == 3:
            return [pow(n, (p + 1) // 4, p)]
        
        s = 0
        q = p - 1
        while q % 2 == 0:
            q //= 2
            s += 1
            
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1
            
        m = s
        c = pow(z, q, p)
        t = pow(n, q, p)
        r = pow(n, (q + 1) // 2, p)
        
        while t != 0 and t != 1:
            t2i = t
            i = 0
            for i in range(1, m):
                t2i = (t2i * t2i) % p
                if t2i == 1:
                    break
            
            b = pow(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p
            
        return [r, p - r]

    def _generate_factor_base(self, size):
        primes = []
        candidate = 2
        while len(primes) < size:
            if isPrime(candidate):
                primes.append(candidate)
            candidate += 1
        return primes

    def build_schnorr_lattice(self):
        """
        Constructs the Schnorr lattice for factoring.
        """
        d = len(self.primes)
        dim = d + 1
        B = np.zeros((dim, dim), dtype=object)

        # Identity part for the exponents e_i
        for i in range(d):
            B[i, i] = 1
            # Weight column
            weight = int(round(self.C * math.log(self.primes[i])))
            B[i, d] = weight

        # The N part
        B[d, d] = int(round(self.C * math.log(self.N)))
        
        return B

    def find_relations(self):
        """
        Use LLL to find smooth relations.
        
        Build a lattice where short vectors correspond to:
        prod(p_i^e_i) * N^e_N ≈ 1
        
        If e_N = 0: We found a smooth number (product of small primes)
        If e_N != 0: We found a relation involving N
        """
        print(f"Starting LLL-based Relation Finding for N={self.N}...")
        print(f"Factor Base Size: {len(self.primes)}")
        
        d = len(self.primes)
        target_relations = d + 20
        
        # Build the Schnorr-style lattice
        # Dimension: d+2 (d primes + N + one extra for scaling)
        # Row i (for prime p_i): [0...1...0 | C*ln(p_i)]
        # Row d+1 (for N):       [0...0...0 | C*ln(N)]
        
        print("Building Schnorr lattice...")
        dim = d + 1
        B = np.zeros((dim, dim), dtype=object)
        
        # Set decimal precision for large number arithmetic
        getcontext().prec = self.precision_bits + 50
        C_dec = Decimal(self.C)
        
        # Identity part for exponents, with log weights
        for i in range(d):
            B[i, i] = 1
            log_p = Decimal(self.primes[i]).ln()
            B[i, d] = int(log_p * C_dec)
        
        # N row
        log_N = Decimal(self.N).ln()
        B[d, d] = int(log_N * C_dec)
        
        print(f"Lattice dimension: {dim}x{dim}")
        print(f"Precision bits (C): {self.precision_bits}")
        
        # Run GeometricLLL reduction (the ACTUAL geometric method)
        print("Running GeometricLLL reduction...")
        lll = GeometricLLL(self.N, basis=B)
        reduced = lll.run_geometric_reduction(verbose=True, num_passes=3)
        
        print("Analyzing reduced basis for relations...")
        
        for row_idx, row in enumerate(reduced):
            # Extract exponents (first d entries) and the weight (last entry)
            exponents = [int(row[i]) for i in range(d)]
            weight = int(row[d])
            
            # Skip zero vector
            if all(e == 0 for e in exponents):
                continue
            
            # Check if this is a valid relation
            # Compute: prod(p_i^e_i) and see if it relates to N
            
            # Separate positive and negative exponents
            pos_product = 1
            neg_product = 1
            
            for i, e in enumerate(exponents):
                if e > 0:
                    pos_product *= pow(self.primes[i], e)
                elif e < 0:
                    neg_product *= pow(self.primes[i], -e)
            
            # Check if pos_product ≡ neg_product (mod N)
            # or pos_product * k = neg_product * N for some small k
            
            if neg_product == 0:
                continue
                
            # Case 1: pos = neg (mod N) gives us x^2 ≡ y^2
            if pos_product % self.N == neg_product % self.N:
                print(f"Row {row_idx}: Found congruence! pos ≡ neg (mod N)")
                
                # This means pos_product - neg_product = k * N
                diff = pos_product - neg_product
                if diff != 0 and diff % self.N == 0:
                    f = GCD(pos_product - 1, self.N)
                    if 1 < f < self.N:
                        print(f"\n[SUCCESS] Factor found: {f}")
                        print(f"Other factor: {self.N // f}")
                        return
            
            # Case 2: Check if pos_product / neg_product is close to N^k
            ratio = pos_product / neg_product if neg_product > 0 else 0
            log_ratio = math.log(ratio) if ratio > 0 else 0
            log_N = math.log(self.N)
            
            k_approx = log_ratio / log_N if log_N > 0 else 0
            
            if abs(k_approx - round(k_approx)) < 0.01 and abs(round(k_approx)) <= 2:
                k = int(round(k_approx))
                print(f"Row {row_idx}: Potential relation with N^{k}")
                print(f"  Exponents (non-zero): {[(self.primes[i], e) for i, e in enumerate(exponents) if e != 0][:10]}...")
                print(f"  Weight: {weight}, Ratio ≈ N^{k_approx:.4f}")
                
                # Store as relation
                # For QS-style, we need x such that x^2 - N is smooth
                # Here we have prod(p^e) ≈ N^k
                
                if k == 1:
                    # prod(p^e) ≈ N means prod(p^e) - N might give us a factor
                    candidate = pos_product if neg_product == 1 else pos_product // neg_product
                    f = GCD(candidate - self.N, self.N)
                    if 1 < f < self.N:
                        print(f"\n[SUCCESS] Factor found: {f}")
                        print(f"Other factor: {self.N // f}")
                        return
                    f = GCD(candidate + self.N, self.N)
                    if 1 < f < self.N:
                        print(f"\n[SUCCESS] Factor found: {f}")
                        print(f"Other factor: {self.N // f}")
                        return
                        
            # Case 3: Standard QS relation - store for linear algebra phase
            # The exponents give us a relation on the primes
            if weight < self.C // 10:  # Small weight means good approximation
                self.relations.append({
                    'x': pos_product % self.N,  # x value
                    'd_exponents': [abs(e) for e in exponents]  # Use absolute values
                })
        
        print(f"Found {len(self.relations)} potential relations from LLL.")
        
        # If LLL alone didn't find enough, do a quick targeted sieve
        # using the short vectors as hints for where to look
        if len(self.relations) < target_relations:
            print(f"\nSupplementing with targeted sieve around sqrt(N)...")
            self._supplementary_sieve(target_relations - len(self.relations))
            
        print(f"Total relations: {len(self.relations)}")

    def _supplementary_sieve(self, needed):
        """Quick sieve to find additional relations if LLL didn't find enough."""
        start_x = math.isqrt(self.N) + 1
        interval_size = min(self.interval_size, 100000)
        
        prime_logs = [math.log(p) for p in self.primes]
        sieve_array = [0.0] * interval_size
        
        # Quick log sieve
        for p_idx, p in enumerate(self.primes):
            log_p = prime_logs[p_idx]
            roots = self._tonelli_shanks(self.N, p)
            for r in roots:
                first_i = (r - start_x) % p
                for i in range(first_i, interval_size, p):
                    sieve_array[i] += log_p
        
        # Get top candidates
        indexed = [(sieve_array[i], i) for i in range(interval_size)]
        indexed.sort(reverse=True)
        
        found = 0
        for score, i in indexed[:needed * 100]:
            x = start_x + i
            val = x * x - self.N
            
            d_exponents = [0] * len(self.primes)
            temp = val
            
            for p_idx, p in enumerate(self.primes):
                while temp % p == 0:
                    d_exponents[p_idx] += 1
                    temp //= p
                if temp == 1: break
            
            if temp == 1:
                self.relations.append({
                    'x': x,
                    'd_exponents': d_exponents
                })
                found += 1
                if found >= needed:
                    break
        
        print(f"  Supplementary sieve found {found} additional relations.")

    def solve_linear_system(self):
        print(f"\nSolving Linear System with {len(self.relations)} relations...")
        if len(self.relations) < len(self.primes) + 5:
            print("Not enough relations to guarantee a solution.")
            return

        # Build Matrix M (relations x primes)
        # We only care about d_exponents mod 2
        
        M = []
        for rel in self.relations:
            row = [x % 2 for x in rel['d_exponents']]
            M.append(row)
            
        # Gaussian Elimination to find Kernel Basis
        matrix = [row[:] for row in M]
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        
        # Augmented with Identity to track combinations
        augmented = []
        for i in range(num_rows):
            aug_row = matrix[i] + [0] * num_rows
            aug_row[num_cols + i] = 1
            augmented.append(aug_row)
            
        pivot_row = 0
        col = 0
        
        while pivot_row < num_rows and col < num_cols:
            pivot_idx = -1
            for i in range(pivot_row, num_rows):
                if augmented[i][col] == 1:
                    pivot_idx = i
                    break
            
            if pivot_idx == -1:
                col += 1
                continue
                
            augmented[pivot_row], augmented[pivot_idx] = augmented[pivot_idx], augmented[pivot_row]
            
            for i in range(num_rows):
                if i != pivot_row and augmented[i][col] == 1:
                    for j in range(col, len(augmented[0])):
                        augmented[i][j] ^= augmented[pivot_row][j]
            
            pivot_row += 1
            col += 1
            
        # Collect Basis Vectors of the Kernel
        kernel_basis = []
        for i in range(num_rows):
            is_zero = True
            for j in range(num_cols):
                if augmented[i][j] != 0:
                    is_zero = False
                    break
            if is_zero:
                kernel_basis.append(augmented[i][num_cols:])
                
        print(f"  Found {len(kernel_basis)} independent dependencies (Kernel size).")
        
        if not kernel_basis:
            print("  No dependencies found.")
            return

        # Randomized Search for Non-Trivial Factors
        import random
        attempts = 0
        max_attempts = 100
        
        print(f"  Attempting to combine dependencies to find X^2 = Y^2 with X != +/-Y...")
        
        while attempts < max_attempts:
            attempts += 1
            
            # Pick a random non-empty subset of the kernel basis
            if len(kernel_basis) > 0:
                mask = [random.randint(0, 1) for _ in range(len(kernel_basis))]
                if sum(mask) == 0: mask[0] = 1
            else:
                break
                
            # Combine the dependency vectors
            combined_dependency = [0] * len(self.relations)
            for k, bit in enumerate(mask):
                if bit:
                    for r in range(len(self.relations)):
                        combined_dependency[r] ^= kernel_basis[k][r]
            
            indices = [k for k, bit in enumerate(combined_dependency) if bit == 1]
            if not indices: continue
            
            # Construct X (product of x_i) and Y (sqrt of product of y_i)
            X = 1
            Y_exponents = [0] * len(self.primes)
            
            for idx in indices:
                rel = self.relations[idx]
                X = (X * rel['x']) % self.N
                for p_idx in range(len(self.primes)):
                    Y_exponents[p_idx] += rel['d_exponents'][p_idx]
            
            # Compute Y = product(p^(exp/2)) mod N
            Y = 1
            valid = True
            
            for p_idx in range(len(self.primes)):
                if Y_exponents[p_idx] % 2 != 0:
                    valid = False
                    break
                    
                y_half = Y_exponents[p_idx] // 2
                Y = (Y * pow(self.primes[p_idx], y_half, self.N)) % self.N
            
            if not valid: continue
                
            # Now X^2 = Y^2 mod N
            # Check if X != +/- Y
            if X == Y or X == (self.N - Y) % self.N:
                # Trivial
                continue
                
            # Check Factors: gcd(X - Y, N)
            f1 = GCD((X - Y) % self.N, self.N)
            f2 = GCD((X + Y) % self.N, self.N)
            
            if f1 > 1 and f1 < self.N:
                print(f"\n[SUCCESS] Factor found: {f1}")
                print(f"Other factor: {self.N // f1}")
                return
            elif f2 > 1 and f2 < self.N:
                print(f"\n[SUCCESS] Factor found: {f2}")
                print(f"Other factor: {self.N // f2}")
                return
        
        print("\n[FAILURE] Could not find non-trivial factors after random attempts.")

if __name__ == "__main__":
    # Target N provided by user
    N = 101010101
    print(f"Target N = {N} ({N.bit_length()} bits)")
    
    # Use 500x500 lattice
    factorizer = GeometricFactorizer(N, lattice_dim=500)
    factorizer.find_relations()
    factorizer.solve_linear_system()
