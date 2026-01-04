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
                # Store FULL exponents for reconstruction, not just parity
                relation_entry = {
                    'x': 1, # RHS is 1 because pos_product == neg_product mod N => pos/neg == 1
                    'exponents': exponents # Full exponents (can be negative)
                }
                
                # PARANOID CHECK: Verify the relation object itself
                check_pos = 1
                check_neg = 1
                for i, e in enumerate(relation_entry['exponents']):
                    if e > 0: check_pos *= pow(self.primes[i], e)
                    elif e < 0: check_neg *= pow(self.primes[i], -e)
                
                if check_pos % self.N != check_neg % self.N:
                    print(f"    [ERROR] Relation sanity check failed! Exponents: {exponents}")
                    continue
                    
                self.relations.append(relation_entry)
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
        
        # We can still try even if we have fewer relations than primes
        # The probability of finding a dependency is lower but not zero
        if len(self.relations) < 10:
            print("Too few relations to attempt solution.")
            return
            
        if len(self.relations) < len(self.primes):
            print(f"Warning: Have {len(self.relations)} relations but {len(self.primes)} primes.")
            print("Attempting anyway - may still find dependencies...")

        # Build Matrix M (relations x primes)
        # We only care about exponents mod 2
        
        M = []
        for rel in self.relations:
            # Use 'exponents' if available (new format), else 'd_exponents' (old format/sieve)
            if 'exponents' in rel:
                row = [abs(x) % 2 for x in rel['exponents']]
            elif 'd_exponents' in rel:
                row = [x % 2 for x in rel['d_exponents']]
            else:
                continue
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
        
        # Verify kernel consistency
        print("  Verifying kernel consistency...")
        valid_kernel_count = 0
        for k_idx, vec in enumerate(kernel_basis):
            # Check if vec * M = 0
            check = [0] * num_cols
            for r, bit in enumerate(vec):
                if bit:
                    for c in range(num_cols):
                        check[c] ^= M[r][c]
            
            if any(x != 0 for x in check):
                print(f"    [ERROR] Kernel vector {k_idx} is INVALID! Residual: {check}")
            else:
                valid_kernel_count += 1
        print(f"  Verified {valid_kernel_count}/{len(kernel_basis)} kernel vectors.")
        
        if not kernel_basis:
            print("  No dependencies found.")
            return

        # Randomized Search for Non-Trivial Factors
        import random
        
        # If kernel is small, try ALL combinations
        if len(kernel_basis) < 16:
            print(f"  Small kernel ({len(kernel_basis)}), trying exhaustive search...")
            max_attempts = 1 << len(kernel_basis)
            use_exhaustive = True
        else:
            print(f"  Large kernel, trying random subsets...")
            max_attempts = 5000
            use_exhaustive = False
        
        print(f"  Attempting to combine dependencies to find X^2 = Y^2 with X != +/-Y...")
        
        trivial_count = 0
        
        for attempt in range(1, max_attempts + 1):
            if use_exhaustive:
                # Generate mask from attempt number
                mask = [(attempt >> k) & 1 for k in range(len(kernel_basis))]
            else:
                # Random mask
                mask = [random.randint(0, 1) for _ in range(len(kernel_basis))]
                if sum(mask) == 0: mask[0] = 1
                
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
                
                if 'exponents' in rel:
                    for p_idx, e in enumerate(rel['exponents']):
                        Y_exponents[p_idx] += e
                elif 'd_exponents' in rel:
                    # Fallback for sieve relations (assumed positive)
                    for p_idx, e in enumerate(rel['d_exponents']):
                        Y_exponents[p_idx] += e
            
            # Compute Y = product(p^(exp/2)) mod N
            Y = 1
            valid = True
            
            for p_idx in range(len(self.primes)):
                if Y_exponents[p_idx] % 2 != 0:
                    valid = False
                    # Debug print for first few failures
                    if attempt < 5:
                        print(f"    Debug: Invalid exponent sum for prime {self.primes[p_idx]}: {Y_exponents[p_idx]}")
                        # Check what the matrix row had
                        # Reconstruct the sum from the matrix rows
                        mat_sum = 0
                        for idx in indices:
                            rel = self.relations[idx]
                            if 'exponents' in rel:
                                mat_sum += abs(rel['exponents'][p_idx]) % 2
                            elif 'd_exponents' in rel:
                                mat_sum += rel['d_exponents'][p_idx] % 2
                        print(f"    Debug: Matrix sum (mod 2) for prime {self.primes[p_idx]}: {mat_sum % 2}")
                    break
                    
                y_half = Y_exponents[p_idx] // 2
                
                base = self.primes[p_idx]
                if y_half < 0:
                    # Handle negative exponent: modular inverse
                    try:
                        base = pow(base, -1, self.N)
                        y_half = -y_half
                    except ValueError:
                        # Base is not invertible -> gcd(base, N) > 1 -> Factor found!
                        f = GCD(base, self.N)
                        print(f"\n[SUCCESS] Factor found during inversion: {f}")
                        print(f"Other factor: {self.N // f}")
                        return
                
                Y = (Y * pow(base, y_half, self.N)) % self.N
            
            if not valid: continue
                
            # Now X^2 = Y^2 mod N
            
            # Debug: Check if X^2 == Y^2
            X2 = pow(X, 2, self.N)
            Y2 = pow(Y, 2, self.N)
            if X2 != Y2:
                if attempt < 5:
                    print(f"    Debug: Relation mismatch! X^2 = {X2}, Y^2 = {Y2}")
                    # Try to diagnose which relation is bad
                    print("    Diagnosing individual relations in this combination...")
                    for idx in indices:
                        rel = self.relations[idx]
                        # Re-verify this relation
                        r_pos = 1
                        r_neg = 1
                        if 'exponents' in rel:
                            for i, e in enumerate(rel['exponents']):
                                if e > 0: r_pos *= pow(self.primes[i], e)
                                elif e < 0: r_neg *= pow(self.primes[i], -e)
                            
                            if r_pos % self.N != r_neg % self.N:
                                print(f"      [BAD RELATION] Index {idx} is invalid! {r_pos%self.N} != {r_neg%self.N}")
                        elif 'd_exponents' in rel:
                            # Sieve relation: x^2 = prod p^e
                            rhs = 1
                            for i, e in enumerate(rel['d_exponents']):
                                rhs *= pow(self.primes[i], e)
                            lhs = pow(rel['x'], 2)
                            if lhs % self.N != rhs % self.N:
                                print(f"      [BAD RELATION] Sieve Index {idx} is invalid!")
            
            # Check if X != +/- Y
            if X == Y or X == (self.N - Y) % self.N:
                trivial_count += 1
                if attempt % 1000 == 0:
                    print(f"    Attempt {attempt}: Found trivial solution (X={X}, Y={Y})...")
                continue
                
            # Check Factors: gcd(X - Y, N)
            f1 = GCD((X - Y) % self.N, self.N)
            f2 = GCD((X + Y) % self.N, self.N)
            
            if attempt < 5:
                print(f"    Debug: Non-trivial candidate! X={X}, Y={Y}")
                print(f"    gcd(X-Y, N) = {f1}")
                print(f"    gcd(X+Y, N) = {f2}")
            
            if f1 > 1 and f1 < self.N:
                print(f"\n[SUCCESS] Factor found: {f1}")
                print(f"Other factor: {self.N // f1}")
                return
            elif f2 > 1 and f2 < self.N:
                print(f"\n[SUCCESS] Factor found: {f2}")
                print(f"Other factor: {self.N // f2}")
                return
        
        print(f"\n[FAILURE] Could not find non-trivial factors after {max_attempts} attempts.")
        print(f"Found {trivial_count} trivial solutions.")

if __name__ == "__main__":
    # Target N provided by user
    N = 261980999226229
    print(f"Target N = {N} ({N.bit_length()} bits)")
    
    # Use 700x700 lattice to get more relations
    factorizer = GeometricFactorizer(N, lattice_dim=700)
    factorizer.find_relations()
    factorizer.solve_linear_system()
