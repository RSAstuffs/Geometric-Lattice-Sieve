import numpy as np
from Crypto.Util.number import getPrime, isPrime, GCD
from geometric_lll import GeometricLLL
import math
import random
from functools import reduce

class GeometricFactorizer:
    def __init__(self, N, factor_base_size=50, precision_bits=100):
        self.N = N
        self.precision_bits = precision_bits
        self.primes = self._generate_factor_base(factor_base_size)
        self.C = 1 << precision_bits  # Scaling factor for logarithms
        self.relations = [] # Store found relations (exponents vector)

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
        print(f"Starting Quadratic Sieve Relation Finding for N={self.N}...")
        print(f"Factor Base Size: {len(self.primes)}")
        
        # Sieve Interval
        start_x = math.isqrt(self.N) + 1
        interval_size = 200000
        sieve_array = [0] * interval_size
        
        # Initialize sieve array with x^2 - N
        print("Initializing sieve array...")
        for i in range(interval_size):
            x = start_x + i
            val = x * x - self.N
            sieve_array[i] = val
            
        # Sieve with factor base
        print("Sieving...")
        for p in self.primes:
            # Solve x^2 = N mod p
            # We need modular square root
            if pow(self.N, (p - 1) // 2, p) != 1:
                continue
                
            # Tonelli-Shanks or simple search for small p
            # Since p is small (factor base), simple search is fine
            roots = []
            for r in range(p):
                if (r * r) % p == (self.N % p):
                    roots.append(r)
            
            for r in roots:
                # Find first index i such that (start_x + i) = r mod p
                # start_x + i = r (mod p) => i = r - start_x (mod p)
                first_i = (r - start_x) % p
                
                # Sieve
                for i in range(first_i, interval_size, p):
                    while sieve_array[i] % p == 0:
                        sieve_array[i] //= p
                        
        # Collect relations
        print("Collecting relations...")
        for i in range(interval_size):
            if sieve_array[i] == 1:
                # Smooth!
                x = start_x + i
                val = x * x - self.N
                
                # Factor val to get exponents
                d_exponents = [0] * len(self.primes)
                temp = val
                for p_idx, p in enumerate(self.primes):
                    while temp % p == 0:
                        d_exponents[p_idx] += 1
                        temp //= p
                
                self.relations.append({
                    'x': x,
                    'd_exponents': d_exponents
                })
                
        print(f"Found {len(self.relations)} smooth relations.")

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
    # Example: 60-bit RSA
    p = getPrime(30)
    q = getPrime(30)
    N = p * q
    print(f"Target N = {N} ({N.bit_length()} bits)")
    
    # Larger factor base for better chance of smoothness
    factorizer = GeometricFactorizer(N, factor_base_size=300, precision_bits=120)
    factorizer.find_relations()
    factorizer.solve_linear_system()
