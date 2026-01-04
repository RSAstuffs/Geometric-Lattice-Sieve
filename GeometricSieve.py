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

    def _factor_over_base(self, n, primes):
        """
        Factors n over the factor base. Returns (exponents, remainder).
        exponents includes -1 as the first element.
        """
        if n == 0: return None, 0
        
        exponents = [0] * (len(primes) + 1) # +1 for -1
        
        if n < 0:
            exponents[0] = 1
            n = -n
            
        for i, p in enumerate(primes):
            while n % p == 0:
                exponents[i+1] += 1
                n //= p
            if n == 1:
                break
                
        return exponents, n

    def find_relations(self):
        """
        Use LLL to find smooth relations.
        """
        print(f"Starting LLL-based Relation Finding for N={self.N}...")
        print(f"Factor Base Size: {len(self.primes)}")
        
        d = len(self.primes)
        target_relations = d + 20
        
        # Set decimal precision
        getcontext().prec = self.precision_bits + 50
        C_dec = Decimal(self.C)
        
        # Precompute logs
        log_primes = [Decimal(p).ln() for p in self.primes]
        log_N = Decimal(self.N).ln()
        
        seen_relations = set()
        
        # Run multiple passes
        max_passes = 50
        
        for pass_num in range(max_passes):
            print(f"\n=== Pass {pass_num + 1} (have {len(self.relations)}/{target_relations} relations) ===")
            
            # Build lattice
            dim = d + 1
            B = np.zeros((dim, dim), dtype=object)
            
            for i in range(d):
                B[i, i] = 1
                B[i, d] = int(log_primes[i] * C_dec)
            
            B[d, d] = int(log_N * C_dec)
            
            # Randomize basis
            if pass_num > 0:
                for _ in range(d):
                    i, j = random.sample(range(d), 2)
                    sign = random.choice([-1, 1])
                    B[i] = B[i] + sign * B[j]
                perm = list(range(d))
                random.shuffle(perm)
                B[:d] = B[perm]
            
            print(f"Lattice dimension: {dim}x{dim}")
            print("Running GeometricLLL reduction...")
            lll = GeometricLLL(self.N, basis=B)
            reduced = lll.run_geometric_reduction(verbose=False, num_passes=3)
            
            print("Analyzing reduced basis for relations...")
            
            pass_relations = 0
            for row in reduced:
                exponents = [int(row[i]) for i in range(d)]
                weight = int(row[d]) # This is approx log(prod) - e_N * log(N)
                
                if all(e == 0 for e in exponents): continue
                
                # Reconstruct e_N from the weight
                # weight ≈ C * (sum e_i ln p_i + e_N ln N)
                # sum e_i ln p_i + e_N ln N ≈ 0
                # e_N ≈ - (sum e_i ln p_i) / ln N
                
                sum_log = sum(e * lp for e, lp in zip(exponents, log_primes))
                e_N = -int(round(sum_log / log_N))
                
                if e_N == 0: continue # Trivial relation between small primes
                
                # Calculate delta
                # If e_N > 0: U * N^e_N - V = delta
                # If e_N < 0: U - V * N^(-e_N) = delta
                
                pos_product = 1
                neg_product = 1
                for i, e in enumerate(exponents):
                    if e > 0: pos_product *= pow(self.primes[i], e)
                    elif e < 0: neg_product *= pow(self.primes[i], -e)
                
                if e_N > 0:
                    delta = pos_product * pow(self.N, e_N) - neg_product
                    # Relation: V = -delta (mod N) -> V * (-delta)^-1 = 1
                    # V is prod p_i^{-e_i} (where e_i < 0)
                    # We need to factor delta
                    delta_exps, remainder = self._factor_over_base(delta, self.primes)
                    
                    if remainder == 1:
                        # Construct relation vector (including -1 at index 0)
                        # V contribution: +(-e_i) for e_i < 0
                        # delta contribution: -delta_exps
                        # -1 contribution: we have -delta, so we need to add 1 to the exponent of -1
                        # (since -delta = (-1) * delta)
                        
                        rel_vec = [0] * (d + 1)
                        # Add V exponents
                        for i, e in enumerate(exponents):
                            if e < 0: rel_vec[i+1] += -e
                            
                        # Subtract delta exponents
                        for i in range(d + 1):
                            rel_vec[i] -= delta_exps[i]
                        
                        # Add the missing -1 factor from V = -delta
                        rel_vec[0] += 1
                            
                        self.relations.append({'vec': rel_vec, 'x': 1})
                        pass_relations += 1
                        
                else: # e_N < 0
                    k = -e_N
                    delta = pos_product - neg_product * pow(self.N, k)
                    # Relation: U = delta (mod N) -> U * delta^-1 = 1
                    delta_exps, remainder = self._factor_over_base(delta, self.primes)
                    
                    if remainder == 1:
                        rel_vec = [0] * (d + 1)
                        # Add U exponents
                        for i, e in enumerate(exponents):
                            if e > 0: rel_vec[i+1] += e
                            
                        # Subtract delta exponents
                        for i in range(d + 1):
                            rel_vec[i] -= delta_exps[i]
                            
                        self.relations.append({'vec': rel_vec, 'x': 1})
                        pass_relations += 1

            print(f"Pass {pass_num + 1}: Found {pass_relations} new relations (total: {len(self.relations)})")
            
            if len(self.relations) >= target_relations:
                break
        
        print(f"Found {len(self.relations)} relations.")
        
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
            
            # Factor val over factor base
            # val = (-1)^s * prod p^e
            
            temp = val
            vec = [0] * (len(self.primes) + 1)
            
            if temp < 0:
                vec[0] = 1
                temp = -temp
            
            for p_idx, p in enumerate(self.primes):
                while temp % p == 0:
                    vec[p_idx+1] += 1
                    temp //= p
                if temp == 1: break
            
            if temp == 1:
                self.relations.append({
                    'x': x,
                    'vec': vec
                })
                found += 1
                if found >= needed:
                    break
        
        print(f"  Supplementary sieve found {found} additional relations.")

    def solve_linear_system(self):
        print(f"\nSolving Linear System with {len(self.relations)} relations...")
        
        if len(self.relations) < len(self.primes):
            print("Warning: Fewer relations than primes.")

        # Build Matrix M (relations x (primes+1)) over GF(2)
        M = []
        for rel in self.relations:
            row = [x % 2 for x in rel['vec']]
            M.append(row)
            
        # Gaussian Elimination
        matrix = [row[:] for row in M]
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        
        print(f"  Building {num_rows}x{num_cols} matrix over GF(2)...")
        
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
            
        kernel_basis = []
        for i in range(num_rows):
            is_zero = True
            for j in range(num_cols):
                if augmented[i][j] != 0:
                    is_zero = False
                    break
            if is_zero:
                kernel_basis.append(augmented[i][num_cols:])
                
        print(f"  Found {len(kernel_basis)} independent dependencies.")
        
        if not kernel_basis:
            return

        print(f"  Trying kernel vectors to find factors...")
        
        import random
        attempts = 0
        tried = set()
        
        while attempts < 1000:
            attempts += 1
            
            if len(kernel_basis) == 1:
                mask = (1,)
            else:
                mask = tuple(random.randint(0, 1) for _ in range(len(kernel_basis)))
                if sum(mask) == 0: mask = (1,) + mask[1:]
            
            if mask in tried: continue
            tried.add(mask)
            
            combined = [0] * len(self.relations)
            for k, bit in enumerate(mask):
                if bit:
                    for r in range(len(self.relations)):
                        combined[r] ^= kernel_basis[k][r]
            
            indices = [k for k, bit in enumerate(combined) if bit == 1]
            if not indices: continue
            
            # Calculate X = prod(x_k)
            X = 1
            for idx in indices:
                X = (X * self.relations[idx]['x']) % self.N
            
            # Calculate Y = sqrt(prod(relations))
            # Each relation is x^2 = (-1)^s * prod p^e (mod N)
            # Or 1 = x^-2 * (-1)^s * prod p^e (mod N)
            # We stored vec = [s, e_1, e_2, ...]
            # Sum of vecs is even (2S, 2E_1, ...)
            # Y = (-1)^S * prod p^E_i
            
            total_exp = [0] * (len(self.primes) + 1)
            for idx in indices:
                vec = self.relations[idx]['vec']
                for i in range(len(total_exp)):
                    total_exp[i] += vec[i]
            
            if any(e % 2 != 0 for e in total_exp): continue
            
            # Y = (-1)^(e_0/2) * prod p_i^(e_i/2)
            Y = 1
            if total_exp[0] // 2 % 2 == 1:
                Y = self.N - 1
                
            for i, p in enumerate(self.primes):
                exp = total_exp[i+1] // 2
                if exp != 0:
                    Y = (Y * pow(p, exp, self.N)) % self.N
            
            # X^2 = Y^2 mod N
            
            if X == Y or X == (self.N - Y) % self.N:
                continue
            
            f1 = GCD(X - Y, self.N)
            if 1 < f1 < self.N:
                print(f"\n[SUCCESS] Factor found: {f1}")
                print(f"Other factor: {self.N // f1}")
                return
                
            f2 = GCD(X + Y, self.N)
            if 1 < f2 < self.N:
                print(f"\n[SUCCESS] Factor found: {f2}")
                print(f"Other factor: {self.N // f2}")
                return
        
        print(f"Failed to find factors after {attempts} attempts.")

if __name__ == "__main__":
    # Target N provided by user
    N = 2021
    print(f"Target N = {N} ({N.bit_length()} bits)")
    
    # Use 700x700 lattice to get more relations
    factorizer = GeometricFactorizer(N, lattice_dim=700)
    factorizer.find_relations()
    factorizer.solve_linear_system()
