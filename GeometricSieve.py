import numpy as np
from Crypto.Util.number import getPrime, isPrime, GCD
from geometric_lll import GeometricLLL
import math
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
        print(f"Building Schnorr Lattice for N={self.N}...")
        print(f"Factor Base Size: {len(self.primes)}")
        
        basis = self.build_schnorr_lattice()
        
        print("Running GeometricLLL Reduction...")
        glll = GeometricLLL(N=self.N, p=1, q=1, basis=basis)
        reduced_basis = glll.run_geometric_reduction(verbose=True)
        
        print("\nAnalyzing Reduced Vectors for Relations...")
        
        # Check each vector in the reduced basis
        for i, vec in enumerate(reduced_basis):
            exponents = vec[:len(self.primes)]
            
            # Calculate the product from exponents
            u = 1
            v = 1
            
            # Reconstruct the relation u/v * N^x_N approx 1
            # We need to find x_N
            current_sum = sum(vec[j] * int(round(self.C * math.log(self.primes[j]))) for j in range(len(self.primes)))
            weight_N = int(round(self.C * math.log(self.N)))
            
            d_idx = len(self.primes)
            remainder = vec[d_idx] - current_sum
            if weight_N == 0: continue
            
            x_N_float = remainder / weight_N
            
            if abs(x_N_float - round(x_N_float)) < 0.001:
                x_N = int(round(x_N_float))
                
                # Calculate u and v
                # Relation is: product(p_i ^ e_i) * N^x_N = 1 + error
                # So product(p_i ^ e_i) = N^(-x_N) + error
                # We want: X^2 = Y^2 mod N
                
                # Let's store the exponent vector for the primes
                # We only care about the exponents modulo 2 for the linear algebra step
                # But we need the full exponents to construct the square root later
                
                # We are looking for product(p_i ^ e_i) = smooth mod N
                # In this lattice, we find u * N^k approx v
                # So u * N^k - v = small_error
                # This means u * N^k = v + small_error
                # If small_error is 0, then u * N^k = v. This is trivial if k=0 and u=v.
                
                # Schnorr's method looks for |u - v*N| < something small
                # Here we have u * N^x_N approx v
                
                # Let's check if u * N^x_N - v is 0 mod N? No.
                # We want X^2 = Y^2 mod N.
                
                # Let's just collect the exponent vectors.
                # If we find a dependency sum(vec_k) = 0 mod 2 (for all components),
                # Then product(relation_k) will have all even exponents.
                # So product(relation_k) = (Something)^2
                
                # The relation is: product(p_i ^ e_i) * N^x_N approx 1
                # This implies product(p_i ^ e_i) is close to a power of N.
                # This isn't exactly X^2 = Y^2 mod N directly.
                
                # Standard Schnorr:
                # Find (e_1, ..., e_d) such that | sum e_i ln p_i - ln N | is small.
                # Then U = product p_i^e_i is close to N.
                # So U - N = small_residue.
                # If small_residue is smooth (factors over factor base), we have a relation:
                # U = N + smooth
                # U = smooth (mod N)
                # product p_i^e_i = product p_j^f_j (mod N)
                
                # Let's try to interpret the vector as U = product p_i^e_i
                # If e_i can be negative, we have U/V approx N.
                # U approx V * N.
                # U - V*N = delta.
                # U = delta (mod N).
                # If delta is smooth, we have U = delta (mod N).
                # U is smooth (by definition). Delta is smooth (hopefully).
                # So we have Smooth1 = Smooth2 (mod N).
                # This is a relation!
                
                # Calculate U and V from positive/negative exponents
                U = 1
                V = 1
                for j, exp in enumerate(exponents):
                    if exp > 0:
                        U *= self.primes[j] ** int(exp)
                    elif exp < 0:
                        V *= self.primes[j] ** int(-exp)
                
                # We have U * N^x_N approx V
                # Let's assume x_N is typically -1 or 1 for simple relations
                # Case 1: U approx V * N  => U - V*N = delta
                # Case 2: U * N approx V  => V - U*N = delta
                
                # Let's calculate the exact integer difference
                # We treat N^x_N term carefully
                
                lhs = U * (self.N ** x_N) if x_N > 0 else U
                rhs = V * (self.N ** (-x_N)) if x_N < 0 else V
                
                # diff = lhs - rhs
                # We want lhs = rhs (mod N) -> this is always true if diff is a multiple of N?
                # No, we want lhs = rhs + delta, where delta is small.
                # Then lhs = delta (mod N) (if rhs is multiple of N? No)
                
                # Wait, the relation we want is X^2 = Y^2 mod N.
                # We build this by multiplying many relations of the form:
                # A_k = B_k mod N
                # where A_k and B_k are smooth.
                
                # In our case:
                # lhs - rhs = delta
                # lhs = rhs + delta
                # lhs = delta (mod N)  (If rhs is a multiple of N? No, rhs is smooth V)
                # lhs = V + delta
                # lhs - V = delta
                # U * N^k - V = delta
                # U * N^k = V + delta
                # Modulo N:
                # If k > 0: 0 = V + delta => V = -delta (mod N)
                # If k = 0: U = V + delta => U - V = delta (mod N)
                # If k < 0: U = V*N^(-k) + delta => U*N^k = V + delta
                
                # Let's look for the case where x_N = 0.
                # Then U approx V. U - V = delta.
                # U = V + delta.
                # U - V = delta.
                # If delta is smooth, we have U/V = delta (mod N)? No.
                
                # Let's look for the case where x_N = 1.
                # U * N approx V.
                # U * N - V = delta.
                # -V = delta (mod N).
                # V = -delta (mod N).
                # V is smooth. If delta is smooth, we have Smooth1 = Smooth2 (mod N).
                
                # Let's calculate delta
                delta = lhs - rhs
                
                # Check if delta is B-smooth (factors over our prime base)
                if delta == 0: continue
                
                delta_abs = abs(delta)
                
                # Factor delta over our factor base
                delta_exponents = [0] * len(self.primes)
                temp_delta = delta_abs
                
                is_smooth = True
                for p_idx, p_val in enumerate(self.primes):
                    while temp_delta % p_val == 0:
                        delta_exponents[p_idx] += 1
                        temp_delta //= p_val
                
                if temp_delta == 1:
                    # Delta is smooth!
                    # We have a relation:
                    # If x_N = 1: V = -delta (mod N)
                    # V is product(p_i ^ v_i)
                    # delta is product(p_i ^ d_i) * (-1 if delta<0)
                    # So product(p_i ^ v_i) = -1 * product(p_i ^ d_i) (mod N)
                    # This is a relation between smooth numbers!
                    
                    # Combine exponents:
                    # V_exponents - Delta_exponents (mod 2)
                    
                    # Let's construct the full exponent vector for this relation
                    # The relation is of the form: product(p_i ^ total_exp_i) = +/- 1 mod N
                    
                    # If x_N = 1:
                    # V = -delta (mod N)
                    # V * delta^-1 = -1 (mod N)
                    # product(p_i ^ v_i) * product(p_i ^ -d_i) = -1 (mod N)
                    # total_exp_i = v_i - d_i
                    
                    # We need to handle the -1. We can treat -1 as a "prime" at index -1.
                    # But for now let's just store the exponent vector.
                    
                    # V exponents come from 'vec' where exp < 0
                    v_exponents = [(-x if x < 0 else 0) for x in exponents]
                    
                    # Total exponents for the relation X = +/- 1 mod N
                    # X = V / delta = V * delta^-1
                    # exp_i = v_exponents[i] - delta_exponents[i]
                    
                    final_exponents = [v_exponents[k] - delta_exponents[k] for k in range(len(self.primes))]
                    
                    # Also track the sign of delta
                    sign = -1 if delta < 0 else 1
                    # If x_N = 1, we have V = -delta => V/delta = -1.
                    # If delta < 0, V/(-|delta|) = -1 => V/|delta| = 1.
                    # If delta > 0, V/|delta| = -1.
                    
                    # We want X^2 = 1 mod N.
                    # We will multiply many relations to get even exponents and +1 sign.
                    
                    print(f"  [+] Found Smooth Relation! Delta={delta}")
                    self.relations.append({
                        'exponents': final_exponents,
                        'sign': -1 if (delta > 0) else 1 # If delta>0, ratio is -1. If delta<0, ratio is 1.
                    })

    def solve_linear_system(self):
        print(f"\nSolving Linear System with {len(self.relations)} relations...")
        if len(self.relations) < len(self.primes):
            print("Not enough relations to guarantee a solution.")
            return

        # Build Matrix M (relations x primes)
        # Include sign column at the start
        M = []
        for rel in self.relations:
            row = [1 if rel['sign'] == -1 else 0] + [x % 2 for x in rel['exponents']]
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
        pivots = []
        
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
            
            pivots.append(col)
            pivot_row += 1
            col += 1
            
        # Collect Basis Vectors of the Kernel
        # Any row in the M part that is all zeros represents a dependency
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
        max_attempts = 20
        
        print(f"  Attempting to combine dependencies to filter trivials...")
        
        while attempts < max_attempts:
            attempts += 1
            
            # Pick a random non-empty subset of the kernel basis
            # This generates a random element of the null space
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
            
            # Construct X
            X_exponents = [0] * len(self.primes)
            for idx in indices:
                rel = self.relations[idx]
                for p_idx, exp in enumerate(rel['exponents']):
                    X_exponents[p_idx] += exp
            
            # Compute X
            X = 1
            for p_idx, total_exp in enumerate(X_exponents):
                half_exp = total_exp // 2
                term = pow(self.primes[p_idx], half_exp, self.N)
                X = (X * term) % self.N
                
            if X == 1 or X == self.N - 1:
                # Trivial
                continue
                
            # Check Factors
            f1 = GCD(X - 1, self.N)
            f2 = GCD(X + 1, self.N)
            
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
    factorizer = GeometricFactorizer(N, factor_base_size=100, precision_bits=120)
    factorizer.find_relations()
    factorizer.solve_linear_system()
