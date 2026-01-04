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
        
        # Factor base size: Use a FIXED small number to ensure relations > primes
        # Key insight: LLL on a (d+1) dimensional lattice gives us d vectors
        # We need relations > primes, so use ~100-150 primes regardless of lattice size
        if factor_base_size is None:
            # For 2048-bit RSA, use ~150 primes
            # LLL will find ~150 relations per pass, with multiple passes we get 300+
            factor_base_size = min(150, lattice_dim - 1)
        
        # Precision bits: should be enough to distinguish log values
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
        print(f"  Factor Base Size: {factor_base_size} primes")
        print(f"  Precision Bits: {precision_bits}")
        print(f"  Target: Find >{factor_base_size} relations to ensure solvable kernel")

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
        target_relations = d + 10  # Need slightly more relations than primes
        
        # Set decimal precision for large number arithmetic
        getcontext().prec = self.precision_bits + 50
        C_dec = Decimal(self.C)
        
        # Precompute logs
        log_primes = [Decimal(p).ln() for p in self.primes]
        log_N = Decimal(self.N).ln()
        
        seen_relations = set()  # Track unique relations
        
        # Run multiple passes with RANDOMIZED bases to get different short vectors
        max_passes = 20
        
        for pass_num in range(max_passes):
            print(f"\n=== Pass {pass_num + 1} (have {len(self.relations)}/{target_relations} relations) ===")
            
            # Build the Schnorr-style lattice
            dim = d + 1
            B = np.zeros((dim, dim), dtype=object)
            
            # Identity part for exponents, with log weights
            for i in range(d):
                B[i, i] = 1
                B[i, d] = int(log_primes[i] * C_dec)
            
            # N row
            B[d, d] = int(log_N * C_dec)
            
            # Apply random unimodular transformation (preserves lattice, changes basis)
            if pass_num > 0:
                # Random row operations: add/subtract rows
                for _ in range(d):
                    i, j = random.sample(range(d), 2)
                    sign = random.choice([-1, 1])
                    B[i] = B[i] + sign * B[j]
                
                # Random permutation of rows (excluding last row which is N)
                perm = list(range(d))
                random.shuffle(perm)
                B[:d] = B[perm]
            
            print(f"Lattice dimension: {dim}x{dim}")
            
            # Run GeometricLLL reduction
            print("Running GeometricLLL reduction...")
            lll = GeometricLLL(self.N, basis=B)
            reduced = lll.run_geometric_reduction(verbose=False, num_passes=3)
            
            print("Analyzing reduced basis for relations...")
            
            pass_relations = 0
            for row_idx, row in enumerate(reduced):
                # Extract exponents (first d entries) and the weight (last entry)
                exponents = tuple(int(row[i]) for i in range(d))
                weight = int(row[d])
                
                # Skip zero vector
                if all(e == 0 for e in exponents):
                    continue
                
                # Normalize: make first non-zero positive (canonical form)
                first_nonzero = next((e for e in exponents if e != 0), 0)
                if first_nonzero < 0:
                    exponents = tuple(-e for e in exponents)
                
                # Skip already seen
                if exponents in seen_relations:
                    continue
                    
                seen_relations.add(exponents)
                
                # Separate positive and negative exponents
                pos_product = 1
                neg_product = 1
                
                for i, e in enumerate(exponents):
                    if e > 0:
                        pos_product *= pow(self.primes[i], e)
                    elif e < 0:
                        neg_product *= pow(self.primes[i], -e)
                
                if neg_product == 0:
                    continue
                    
                # Quick factor check
                if pos_product % self.N == neg_product % self.N:
                    diff = pos_product - neg_product
                    if diff != 0 and diff % self.N == 0:
                        f = GCD(pos_product - 1, self.N)
                        if 1 < f < self.N:
                            print(f"\n[SUCCESS] Factor found: {f}")
                            print(f"Other factor: {self.N // f}")
                            return
                
                # Store as relation for linear algebra phase
                # Accept all non-trivial relations
                self.relations.append({
                    'x': pos_product % self.N,
                    'd_exponents': [abs(e) % 2 for e in exponents]  # GF(2) parity
                })
                pass_relations += 1
            
            print(f"Pass {pass_num + 1}: Found {pass_relations} new relations (total: {len(self.relations)})")
            
            if len(self.relations) >= target_relations:
                print(f"Target reached! Have {len(self.relations)} relations > {d} primes.")
                break
        
        print(f"Found {len(self.relations)} relations from LLL passes.")
        
        # If LLL alone didn't find enough, do a targeted sieve
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
        
        if len(self.relations) < 10:
            print("Too few relations to attempt solution.")
            return
            
        if len(self.relations) < len(self.primes):
            print(f"Warning: Have {len(self.relations)} relations but {len(self.primes)} primes.")
            print("Attempting anyway - may still find dependencies...")

        # Build Matrix M (relations x primes) over GF(2)
        M = []
        for rel in self.relations:
            row = [x % 2 for x in rel['d_exponents']]
            M.append(row)
            
        # Gaussian Elimination to find Kernel Basis
        matrix = [row[:] for row in M]
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        
        print(f"  Building {num_rows}x{num_cols} matrix over GF(2)...")
        
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

        # For LLL-based relations, we have: prod(p_i^e_i) ≈ 1
        # When we combine relations with even total exponents, we get:
        # prod(p_i^{2*f_i}) = (prod(p_i^f_i))^2 = Y^2
        # 
        # We need to find X such that X^2 ≡ Y^2 (mod N)
        # Strategy: Use the combined product directly and compute GCDs
        
        print(f"  Trying {len(kernel_basis)} kernel vectors to find factors...")
        
        # Try each kernel vector and combinations
        max_attempts = min(1000, len(kernel_basis) * 50)
        attempts = 0
        tried_combinations = set()
        
        while attempts < max_attempts:
            attempts += 1
            
            # Pick a random non-empty subset of the kernel basis
            if len(kernel_basis) == 1:
                mask = (1,)
            else:
                mask = tuple(random.randint(0, 1) for _ in range(len(kernel_basis)))
                if sum(mask) == 0:
                    mask = (1,) + mask[1:]
            
            if mask in tried_combinations:
                continue
            tried_combinations.add(mask)
                
            # Combine the dependency vectors
            combined = [0] * len(self.relations)
            for k, bit in enumerate(mask):
                if bit:
                    for r in range(len(self.relations)):
                        combined[r] ^= kernel_basis[k][r]
            
            indices = [k for k, bit in enumerate(combined) if bit == 1]
            if not indices:
                continue
            
            # Compute total exponents when combining these relations
            total_exp = [0] * len(self.primes)
            for idx in indices:
                rel = self.relations[idx]
                for p_idx in range(len(self.primes)):
                    total_exp[p_idx] += rel['d_exponents'][p_idx]
            
            # All exponents should be even (that's what kernel means)
            if any(e % 2 != 0 for e in total_exp):
                continue
            
            # Compute Y = prod(p_i^(exp_i/2)) mod N
            Y = 1
            for p_idx, exp in enumerate(total_exp):
                half_exp = exp // 2
                if half_exp > 0:
                    Y = (Y * pow(self.primes[p_idx], half_exp, self.N)) % self.N
            
            # For LLL relations, the product prod(p^e) ≈ 1, so:
            # prod(p^e) = 1 + k*N for some small k (or = N^m / (1 + k*N))
            # 
            # Compute X as product of x values (which are prod_pos % N)
            X = 1
            for idx in indices:
                X = (X * self.relations[idx]['x']) % self.N
            
            # Try various factor extractions
            candidates = [
                (X - Y) % self.N,
                (X + Y) % self.N,
                (X - 1) % self.N,
                (X + 1) % self.N,
                (Y - 1) % self.N,
                (Y + 1) % self.N,
                (X * Y - 1) % self.N,
                (X * Y + 1) % self.N,
            ]
            
            # Also try X*Y^(-1) - 1 if Y is invertible
            try:
                Y_inv = pow(Y, -1, self.N)
                candidates.append((X * Y_inv - 1) % self.N)
                candidates.append((X * Y_inv + 1) % self.N)
            except:
                pass
            
            for cand in candidates:
                if cand == 0:
                    continue
                f = GCD(cand, self.N)
                if 1 < f < self.N:
                    print(f"\n[SUCCESS] Factor found: {f}")
                    print(f"Other factor: {self.N // f}")
                    return
        
        print(f"\n[FAILURE] Could not find non-trivial factors after {attempts} attempts.")
        print("  This may indicate the relations aren't capturing the structure of N.")
        print("  Try increasing lattice_dim or running again (randomized algorithm).")

if __name__ == "__main__":
    # Target N provided by user
    N = 1212
    print(f"Target N = {N} ({N.bit_length()} bits)")
    
    # Use 700x700 lattice to get more relations
    factorizer = GeometricFactorizer(N, lattice_dim=700)
    factorizer.find_relations()
    factorizer.solve_linear_system()
