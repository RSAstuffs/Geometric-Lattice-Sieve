import numpy as np
from Crypto.Util.number import getPrime, isPrime, GCD
from geometric_lll import GeometricLLL
from RelationTransformer import RelationTransformer
import math
import random
import copy
from functools import reduce
from decimal import Decimal, getcontext
from scipy.spatial import Delaunay

def jacobi_symbol(a, n):
    """
    Calculates the Jacobi symbol (a/n).
    n must be a positive odd integer.
    """
    if n <= 0: return 0
    if n % 2 == 0: return 0
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    if n == 1:
        return result
    else:
        return 0

class GeometricFactorizer:
    def __init__(self, N, factor_base_size=None, precision_bits=None, interval_size=None, lattice_dim=500):
        self.N = N
        self.lattice_dim = lattice_dim
        
        # Scale parameters based on bit length of N
        n_bits = N.bit_length()
        
        # Factor base size: Use lattice_dim - 1 primes to match the lattice dimension
        # Key insight: LLL on a (d+1) dimensional lattice gives us d vectors
        # We need relations > primes, so factor_base_size = lattice_dim - 1
        if factor_base_size is None:
            # Factor base size is lattice_dim - 1 (adjustable via GUI)
            factor_base_size = lattice_dim - 1
        
        # Precision bits: should be enough to distinguish log values
        if precision_bits is None:
            precision_bits = max(100, n_bits + 50)
        
        # Interval size for supplementary sieve
        if interval_size is None:
            interval_size = max(100000, min(100000000, factor_base_size * 10000))
        
        self.precision_bits = precision_bits
        self.interval_size = interval_size
        self.primes = self._generate_factor_base(factor_base_size)
        self.C = 1 << precision_bits  # Scaling factor for logarithms
        self.relations = []
        self.partial_relations = {} # Store partial relations (Large Primes)
        
        print(f"N bit length: {n_bits}")
        print(f"Lattice dimension: {lattice_dim}x{lattice_dim}")
        print(f"Auto-scaled parameters:")
        print(f"  Factor Base Size: {factor_base_size} primes")
        print(f"  Precision Bits: {precision_bits}")
        print(f"  Target: Find >{factor_base_size} relations to ensure solvable kernel")
        
        # Warn and auto-adjust if lattice dimension is too small
        # Recommended lattice dimensions based on N size:
        # For reliable factorization, need larger dimensions than README suggests
        # README values may work in special cases but are not generally sufficient
        if n_bits < 64:
            min_lattice_dim = 50
        elif n_bits < 128:
            min_lattice_dim = 100
        elif n_bits < 256:
            min_lattice_dim = 200
        elif n_bits < 512:
            min_lattice_dim = 500
        elif n_bits < 1024:
            min_lattice_dim = 2000  # More aggressive for 1024-bit RSA
        elif n_bits < 2048:
            min_lattice_dim = 3000  # For 2048-bit RSA, need much larger
        else:
            # For very large N (>= 2048 bits), scale aggressively
            # Formula: roughly n_bits * 1.5 to 2.0 for reliable results
            min_lattice_dim = max(5000, int(n_bits * 1.5))
        
        if lattice_dim < min_lattice_dim:
            print(f"  [WARNING] Lattice dimension ({lattice_dim}) is too small for {n_bits}-bit N!")
            print(f"  Recommended minimum: {min_lattice_dim}×{min_lattice_dim}")
            if n_bits >= 2048:
                print(f"  For 2048-bit N: 3000-5000 is recommended for reliable factorization.")
                print(f"  (README suggests 1000, but this only works in special cases)")
            print(f"  Auto-adjusting to minimum recommended: {min_lattice_dim}")
            lattice_dim = min_lattice_dim
            self.lattice_dim = lattice_dim
            # Recalculate factor base size
            factor_base_size = lattice_dim - 1
            self.primes = self._generate_factor_base(factor_base_size)
            print(f"  Updated: Factor Base Size = {factor_base_size} primes")

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
        # Always include 2
        primes.append(2)
        candidate = 3
        while len(primes) < size:
            if isPrime(candidate):
                primes.append(candidate)
            candidate += 2
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

    def _point_in_triangle(self, point, triangle):
        """Check if a point is inside a triangle using barycentric coordinates."""
        p = point
        a, b, c = triangle
        
        # Compute barycentric coordinates
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-10:
            return False  # Degenerate triangle
        
        alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
        beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
        gamma = 1 - alpha - beta
        
        return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1

    def _distance_to_triangle(self, point, triangle):
        """Compute minimum distance from point to triangle."""
        p = point
        a, b, c = triangle
        
        # Check distance to each edge
        edges = [(a, b), (b, c), (c, a)]
        min_dist = float('inf')
        
        for edge_start, edge_end in edges:
            dist = self._point_to_line_distance(p, edge_start, edge_end)
            min_dist = min(min_dist, dist)
        
        return min_dist

    def _point_to_line_distance(self, point, line_start, line_end):
        """Compute distance from point to line segment."""
        p = np.array(point)
        a = np.array(line_start)
        b = np.array(line_end)
        
        # Vector from a to b
        ab = b - a
        # Vector from a to p
        ap = p - a
        
        # Projection scalar
        proj = np.dot(ap, ab)
        len2 = np.dot(ab, ab)
        
        if len2 == 0:
            return np.linalg.norm(p - a)
        
        # Clamp projection to line segment
        t = max(0, min(1, proj / len2))
        
        # Closest point on line segment
        closest = a + t * ab
        
        return np.linalg.norm(p - closest)

    def _legendre_symbol(self, a, p):
        """Compute Legendre symbol (a/p)."""
        if a == 0:
            return 0
        # Use Euler's criterion: (a/p) ≡ a^((p-1)/2) mod p
        return pow(a, (p - 1) // 2, p)

    def _modular_square_root(self, a, p):
        """Find square root of a mod p using Tonelli-Shanks algorithm."""
        if a == 0:
            return 0

        # Check if a is a quadratic residue
        if self._legendre_symbol(a, p) != 1:
            return None

        # For p = 2
        if p == 2:
            return a % 2

        # Write p-1 = 2^s * q
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1

        # Find a non-quadratic residue z
        z = 2
        while self._legendre_symbol(z, p) != -1:
            z += 1

        # Initialize
        m = s
        c = pow(z, q, p)
        t = pow(a, q, p)
        r = pow(a, (q + 1) // 2, p)

        while True:
            if t == 0:
                return 0
            if t == 1:
                return r

            # Find smallest i such that t^(2^i) ≡ 1 mod p
            i = 1
            t2i = pow(t, 2, p)
            while t2i != 1:
                t2i = pow(t2i, 2, p)
                i += 1

            # Update
            b = pow(c, 1 << (m - i - 1), p)
            r = (r * b) % p
            c = (b * b) % p
            t = (t * c) % p
            m = i

    def _solve_quadratic_congruence(self, a, n):
        """Solve x² ≡ a mod n."""
        # For now, use a simple approach: try small x values
        # In practice, this would use more sophisticated methods
        for x in range(2, min(1000, n)):
            if (x * x) % n == a % n:
                return x
        return None

    def _is_smooth(self, n, factor_base):
        """Check if n is smooth over the factor base."""
        if n == 1:
            return True
        for p in factor_base:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
        return n == 1

    def _factor_over_base(self, n, factor_base):
        """Factor n over the factor base and return exponents and remainder."""
        exponents = [0] * len(factor_base)
        original_n = n
        n = abs(n)
        for i, p in enumerate(factor_base):
            while n % p == 0:
                exponents[i] += 1
                n //= p
        # Compute the product of factored primes
        prod = 1
        for i, e in enumerate(exponents):
            if e > 0:
                prod *= factor_base[i] ** e
        remainder = original_n // prod
        return exponents, remainder

    def find_relations(self):
        """
        Use LLL to find smooth relations.
        """
        print(f"Starting LLL-based Relation Finding for N={self.N}...")
        print(f"Factor Base Size: {len(self.primes)}")
        
        d = len(self.primes)
        target_relations = d + 200
        
        getcontext().prec = self.precision_bits + 50
        C_dec = Decimal(self.C)
        
        log_primes = [Decimal(p).ln() for p in self.primes]
        log_N = Decimal(self.N).ln()
        
        # Store for transformer use
        self._d = d
        self._log_primes = log_primes
        self._log_N = log_N
        self._C_dec = C_dec
        
        seen_relations = set()
        
        # Store successful coefficients for training the Transformer
        self.successful_coeffs = []  # Smooth relations (remainder = ±1)
        self.failed_coeffs = []  # Partial relations (large prime) - learn what NOT to do
        
        max_passes = 100
        
        for pass_num in range(max_passes):
            print(f"\n=== Pass {pass_num + 1} (have {len(self.relations)}/{target_relations} relations) ===")
            
            dim = d + 1
            B = np.zeros((dim, dim), dtype=object)
            
            for i in range(d):
                B[i, i] = 1
                B[i, d] = int(log_primes[i] * C_dec)
            
            B[d, d] = int(log_N * C_dec)
            
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
            # Use more passes to find better relations (especially smooth ones)
            # More passes = better reduction = more likely to find smooth relations
            num_passes = 3 if pass_num < 5 else 1  # More passes early, fewer later
            reduced = lll.run_geometric_reduction(verbose=False, num_passes=num_passes)
            
            print("Analyzing reduced basis for relations...")
            
            pass_relations = 0
            stats_smooth = 0  # Count smooth relations found
            stats_partial = 0  # Count partial relations found
            
            # Helper to process a vector of exponents
            def process_vector(exponents, coeffs_record=None, basis_snapshot=None):
                nonlocal pass_relations, stats_smooth, stats_partial
                was_smooth = False  # Will be set based on remainder
                
                if all(e == 0 for e in exponents): return
                
                # Reconstruct e_N
                sum_log = sum(e * lp for e, lp in zip(exponents, log_primes))
                e_N = -int(round(sum_log / log_N))
                
                if e_N == 0: return 
                
                # Calculate delta
                pos_product = 1
                neg_product = 1
                for i, e in enumerate(exponents):
                    if e > 0: pos_product *= pow(self.primes[i], e)
                    elif e < 0: neg_product *= pow(self.primes[i], -e)
                
                delta = 0
                if e_N > 0:
                    delta = pos_product * pow(self.N, e_N) - neg_product
                else:
                    delta = pos_product - neg_product * pow(self.N, -e_N)
                
                if delta == 0: return

                # Factor delta
                delta_exps, remainder = self._factor_over_base(delta, self.primes)
                
                # Debug: Track remainder sizes to understand why we're not getting smooth relations
                if pass_num == 0 and stats_smooth + stats_partial < 100:
                    abs_remainder = abs(remainder)
                    if abs_remainder > 1:
                        # Log some examples of remainders to understand the distribution
                        if stats_partial % 50 == 0:
                            print(f"      Example: delta remainder = {abs_remainder} (bits: {abs_remainder.bit_length()})")
                
                # Track if this was smooth (remainder = ±1) for transformer learning
                was_smooth = (remainder == 1 or remainder == -1)
                
                if was_smooth:
                    stats_smooth += 1
                    # Construct DOUBLE-WIDTH relation vector
                    # We want to find a subset where:
                    #   prod(LHS_i) = X^2
                    #   prod(RHS_i) = Y^2
                    # Then X^2 = Y^2 mod N
                    # LHS is neg_product (or pos_product), RHS is delta (or -delta)
                    
                    # Vector structure: [LHS_sign, LHS_exponents..., RHS_sign, RHS_exponents...]
                    # Length: 1 + d + 1 + d = 2d + 2
                    
                    lhs_vec = [0] * (d + 1)
                    rhs_vec = [0] * (d + 1)
                    
                    if e_N > 0:
                        # neg_product = -delta (mod N)
                        # LHS = neg_product, RHS = -delta
                        
                        # Fill LHS (neg_product)
                        for i, e in enumerate(exponents):
                            if e < 0: lhs_vec[i+1] += -e
                            
                        # Fill RHS (-delta)
                        rhs_vec = [0] * (d + 1)
                        rhs_vec[0] = 1  # sign for -delta
                        rhs_vec[1:] = delta_exps
                        
                    else:
                        # pos_product = delta (mod N)
                        # LHS = pos_product, RHS = delta
                        
                        # Fill LHS (pos_product)
                        for i, e in enumerate(exponents):
                            if e > 0: lhs_vec[i+1] += e
                            
                        # Fill RHS (delta)
                        rhs_vec = [0] * (d + 1)
                        rhs_vec[0] = 0
                        rhs_vec[1:] = delta_exps

                    # Normalize signs
                    lhs_vec[0] %= 2
                    rhs_vec[0] %= 2
                    
                    # Combine into one vector for linear algebra
                    full_vec = lhs_vec + rhs_vec
                    
                    # Verify relation: LHS == RHS (mod N)
                    # This is just a sanity check that the relation is valid
                    # It doesn't check for squares yet
                    
                    valid_rel = True
                    
                    # Check for triviality: If LHS == 1 or RHS == 1, it's a trivial relation
                    # LHS == 1 means all exponents are 0 (except maybe sign)
                    # But we want to avoid X=1, which means LHS=1 (since X comes from LHS)
                    
                    is_lhs_one = all(e == 0 for e in lhs_vec[1:])
                    is_rhs_one = all(e == 0 for e in rhs_vec[1:])
                    
                    if is_lhs_one or is_rhs_one:
                        # print("    Debug: Skipping trivial relation (X=1 or Y=1)")
                        valid_rel = False
                    
                    # Check if LHS == RHS (X == Y) - this would make the relation useless
                    if lhs_vec == rhs_vec:
                        valid_rel = False
                        if stats_smooth < 5:  # Debug first few
                            print(f"      Rejected: LHS == RHS (would give X == Y)")
                    
                    # Check if LHS == -RHS (X == -Y mod N) - would lead to trivial factorization
                    # This happens when signs differ but exponents are the same
                    lhs_neg = lhs_vec[:]
                    lhs_neg[0] = (lhs_neg[0] + 1) % 2  # Flip sign
                    if lhs_neg == rhs_vec:
                        valid_rel = False
                        if stats_smooth < 5:  # Debug first few
                            print(f"      Rejected: LHS == -RHS (would give X == -Y)")
                    
                    # Check if relation is all zeros (completely trivial)
                    if all(e == 0 for e in full_vec):
                        valid_rel = False
                    
                    # Additional check: Ensure LHS and RHS are actually different
                    # If they're too similar, the relation won't help
                    lhs_nonzero = sum(1 for e in lhs_vec[1:] if e != 0)
                    rhs_nonzero = sum(1 for e in rhs_vec[1:] if e != 0)
                    if lhs_nonzero == 0 or rhs_nonzero == 0:
                        valid_rel = False
                        if stats_smooth < 5:
                            print(f"      Rejected: LHS or RHS has no non-zero exponents")
                        
                    if valid_rel:
                        self.relations.append({
                            'type': 'double',
                            'vec': full_vec
                        })
                        pass_relations += 1
                        if coeffs_record is not None and basis_snapshot is not None:
                            if was_smooth:
                                self.successful_coeffs.append((basis_snapshot, coeffs_record))
                            else:
                                # Store failed attempts too - learn what NOT to do
                                self.failed_coeffs.append((basis_snapshot, coeffs_record))
                
                elif jacobi_symbol(self.N, abs(remainder)) == 1:
                    stats_partial += 1
                    # Large Prime Variation: Only accept if we've seen this remainder before
                    # This ensures each extra factor column has at least 2 relations
                    lp = abs(remainder)
                    
                    # Check if we already have a relation with this remainder
                    if lp not in self.partial_relations:
                        # Store for later - don't add to relations yet
                        rel_vec = [0] * (d + 1)
                        X_val = 1
                        
                        if e_N > 0:
                            for i, e in enumerate(exponents):
                                if e < 0: rel_vec[i+1] += -e
                            for i in range(d):
                                rel_vec[i+1] -= delta_exps[i]
                            rel_vec[0] -= 1 
                        else:
                            for i, e in enumerate(exponents):
                                if e > 0: rel_vec[i+1] += e
                            for i in range(d):
                                rel_vec[i+1] -= delta_exps[i]
                        
                        rel_vec[0] = rel_vec[0] % 2
                        
                        self.partial_relations[lp] = {
                            'x': X_val,
                            'exponents': rel_vec,
                            'remainder': lp
                        }
                    else:
                        # We have a match! Add both relations
                        rel_vec = [0] * (d + 1)
                        X_val = 1
                        
                        if e_N > 0:
                            for i, e in enumerate(exponents):
                                if e < 0: rel_vec[i+1] += -e
                            for i in range(d):
                                rel_vec[i+1] -= delta_exps[i]
                            rel_vec[0] -= 1 
                        else:
                            for i, e in enumerate(exponents):
                                if e > 0: rel_vec[i+1] += e
                            for i in range(d):
                                rel_vec[i+1] -= delta_exps[i]
                        
                        rel_vec[0] = rel_vec[0] % 2
                        
                        # Add the stored relation first (if not already added)
                        stored = self.partial_relations.pop(lp)
                        self.relations.append(stored)
                        
                        # Add the new relation
                        self.relations.append({
                            'x': X_val,
                            'exponents': rel_vec,
                            'remainder': lp
                        })
                        pass_relations += 2
                        # Large prime matches are also useful (they become legacy, but still valid)
                        if coeffs_record is not None and basis_snapshot is not None:
                            # These are partial relations that matched - store as "failed" pattern
                            # (we want smooth, not partial)
                            self.failed_coeffs.append((basis_snapshot, coeffs_record))

            # 1. Check basis vectors
            basis_vectors = []
            reduced_snapshot = copy.deepcopy(reduced)
            for r_idx, row in enumerate(reduced):
                vec = [int(row[i]) for i in range(d)]
                basis_vectors.append(vec)
                
                # Record coeffs: just 1 at r_idx
                coeffs = np.zeros(dim)
                coeffs[r_idx] = 1
                
                process_vector(vec, coeffs, reduced_snapshot)
            
            # 2. Lattice Sieving: Check random linear combinations
            # This generates many more short vectors from the reduced basis
            # Increase combinations if we're not finding enough relations overall
            num_combinations = 20000 if len(self.relations) < target_relations // 2 else 10000
            print(f"    Sieving lattice (checking {num_combinations} combinations)...")
            for _ in range(num_combinations):
                # Pick 2-3 vectors
                k = random.randint(2, 3)
                indices = random.sample(range(len(basis_vectors)), k)
                
                # Combine
                new_vec = [0] * d
                coeffs = np.zeros(dim)
                
                for idx in indices:
                    coeff = random.choice([-1, 1])
                    coeffs[idx] = coeff
                    for i in range(d):
                        new_vec[i] += coeff * basis_vectors[idx][i]
                
                process_vector(new_vec, coeffs, reduced_snapshot)
            
            print(f"Pass {pass_num + 1}: Found {pass_relations} new relations (total: {len(self.relations)})")
            if pass_num == 0 or pass_relations > 0:
                print(f"  Statistics: {stats_smooth} smooth (remainder=±1), {stats_partial} partial (large prime)")
            
            # If we're not finding smooth relations after several passes, run sieve early
            # Smooth relations are critical - without them we only get legacy relations
            non_legacy_count = sum(1 for r in self.relations 
                                  if ('type' in r and r['type'] == 'double') or 'd_exponents' in r)
            
            # After 3 passes, if we have < 10 smooth relations, run sieve to get usable relations
            if pass_num >= 3 and stats_smooth < 10 and non_legacy_count < 20:
                print(f"  [WARNING] Very few smooth relations found ({stats_smooth} this pass).")
                print(f"  Running supplementary sieve early to find usable relations...")
                needed = max(50, target_relations - len(self.relations))
                self._supplementary_sieve(min(needed, 100))  # Limit to avoid long waits
            
            if len(self.relations) >= target_relations:
                print(f"Target reached! Have {len(self.relations)} relations > {d} primes.")
                break
        
        print(f"Found {len(self.relations)} relations from LLL passes.")
        
        # Count relation types
        double_count = sum(1 for r in self.relations if 'type' in r and r['type'] == 'double')
        sieve_count = sum(1 for r in self.relations if 'd_exponents' in r)
        legacy_count = sum(1 for r in self.relations if 'exponents' in r and 'type' not in r)
        print(f"  Relation breakdown: {double_count} double, {sieve_count} sieve, {legacy_count} legacy")
        
        # Store the last reduced basis for transformer use
        self.last_reduced_basis = None
        if 'reduced' in locals():
            self.last_reduced_basis = reduced
        
        # --- Transformer Learning & Divining ---
        # Count smooth relations needed
        d = self._d  # Use stored dimension
        smooth_count = sum(1 for r in self.relations 
                          if ('type' in r and r['type'] == 'double') or 
                             ('d_exponents' in r and r.get('remainder', 1) == 1))
        min_smooth_needed = max(50, d // 2)  # Need enough smooth relations
        
        # Train transformer on both successful (smooth) and failed (partial) attempts
        total_training_samples = len(self.successful_coeffs) + len(self.failed_coeffs)
        
        if total_training_samples > 0:
            print(f"\n=== Transformer Learning & Divining ===")
            print(f"Training on {len(self.successful_coeffs)} smooth + {len(self.failed_coeffs)} partial patterns...")
            print(f"Current smooth relations: {smooth_count}/{min_smooth_needed} needed")
            
            # Initialize model
            model = RelationTransformer(self._d + 1, model_dim=64, num_heads=4, num_layers=2)
            
            # Train on successful patterns (smooth relations) - these are GOOD
            for i, (basis, coeffs) in enumerate(self.successful_coeffs):
                basis_obj = np.array(basis, dtype=object)
                max_val = np.max(np.abs(basis_obj))
                if max_val == 0: max_val = 1
                basis_mat = (basis_obj / max_val).astype(np.float64)
                # Train with positive reinforcement (these led to smooth relations)
                model.train(basis_mat, [coeffs], iterations=15, positive=True)
            
            # Train on failed patterns (partial relations) - learn to AVOID these
            for i, (basis, coeffs) in enumerate(self.failed_coeffs[:len(self.successful_coeffs)]):  # Balance dataset
                basis_obj = np.array(basis, dtype=object)
                max_val = np.max(np.abs(basis_obj))
                if max_val == 0: max_val = 1
                basis_mat = (basis_obj / max_val).astype(np.float64)
                # Train with negative reinforcement (these led to partial relations)
                # We want to learn to avoid patterns similar to these
                model.train(basis_mat, [coeffs], iterations=10, positive=False)
            
            print(f"  Training complete. Divining smooth relations...")
            
            # Use the FINAL reduced basis from the last pass
            if self.last_reduced_basis is not None:
                reduced_obj = np.array(self.last_reduced_basis, dtype=object)
            else:
                # Fallback: rebuild a basis from current relations if needed
                print(f"  [WARNING] No reduced basis available, rebuilding...")
                # Re-run one LLL pass to get a basis
                d = self._d
                log_primes = self._log_primes
                log_N = self._log_N
                C_dec = self._C_dec
                dim = d + 1
                B = np.zeros((dim, dim), dtype=object)
                for i in range(d):
                    B[i, i] = 1
                    B[i, d] = int(log_primes[i] * C_dec)
                B[d, d] = int(log_N * C_dec)
                lll = GeometricLLL(self.N, basis=B)
                reduced_obj = np.array(lll.run_geometric_reduction(verbose=False, num_passes=1), dtype=object)
            
            max_val = np.max(np.abs(reduced_obj))
            if max_val == 0: max_val = 1
            final_basis = (reduced_obj / max_val).astype(np.float64)
            
            # Keep divining until we have enough smooth relations
            transformer_attempts = 0
            max_transformer_attempts = 10000  # Safety limit
            transformer_found = 0
            
            while smooth_count < min_smooth_needed and transformer_attempts < max_transformer_attempts:
                transformer_attempts += 1
                
                # Divine coefficients that might lead to smooth relations
                pred_coeffs = model.divine_coefficients(final_basis)
                
                new_vec = [0] * d
                is_zero = True
                for i in range(d):
                    if pred_coeffs[i] != 0:
                        is_zero = False
                        for j in range(d):
                            new_vec[j] += pred_coeffs[i] * final_basis[i][j]
                
                if not is_zero:
                    # Convert to integers
                    new_vec = [int(round(x)) for x in new_vec]
                    # Process and check if it was smooth
                    old_smooth = smooth_count
                    process_vector(new_vec, None, None)
                    # Re-count smooth relations
                    smooth_count = sum(1 for r in self.relations 
                                      if ('type' in r and r['type'] == 'double') or 
                                         ('d_exponents' in r and r.get('remainder', 1) == 1))
                    
                    if smooth_count > old_smooth:
                        transformer_found += 1
                        if transformer_found % 10 == 0:
                            print(f"  Transformer found {transformer_found} smooth relations ({smooth_count}/{min_smooth_needed})...")
                
                # Update model with new patterns if we found something
                if transformer_attempts % 100 == 0 and len(self.successful_coeffs) > len(self.failed_coeffs):
                    # Retrain periodically with new data
                    if len(self.successful_coeffs) > 0:
                        latest_basis = self.successful_coeffs[-1][0]
                        latest_coeffs = self.successful_coeffs[-1][1]
                        basis_obj = np.array(latest_basis, dtype=object)
                        max_val = np.max(np.abs(basis_obj))
                        if max_val == 0: max_val = 1
                        basis_mat = (basis_obj / max_val).astype(np.float64)
                        model.train(basis_mat, [latest_coeffs], iterations=5, positive=True)
            
            if smooth_count >= min_smooth_needed:
                print(f"  ✓ Transformer found enough smooth relations! ({smooth_count}/{min_smooth_needed})")
            else:
                print(f"  Transformer phase complete. Found {transformer_found} smooth relations ({smooth_count}/{min_smooth_needed})")
        elif len(self.successful_coeffs) == 0:
            print(f"\n[WARNING] No successful patterns to train transformer on.")
            print(f"  Need to find at least some smooth relations first.")
        # ---------------------------------------
        
        # Count non-legacy relations (double + sieve)
        non_legacy_count = sum(1 for r in self.relations 
                              if ('type' in r and r['type'] == 'double') or 'd_exponents' in r)
        
        # If we don't have enough non-legacy relations, run supplementary methods
        # Even if we have enough total relations, we need non-legacy ones for factorization
        min_non_legacy = max(50, d // 2)  # Need at least some non-legacy relations
        
        if non_legacy_count < min_non_legacy:
            needed = max(target_relations - len(self.relations), min_non_legacy - non_legacy_count)
            print(f"\nNeed more non-legacy relations ({non_legacy_count}/{min_non_legacy}).")
            print(f"Supplementing with targeted sieve around sqrt(N)...")
            self._supplementary_sieve(needed)
            
            # Re-count after sieve
            non_legacy_count = sum(1 for r in self.relations 
                                  if ('type' in r and r['type'] == 'double') or 'd_exponents' in r)
        
        # If still not enough non-legacy, try Dixon's random squares
        if non_legacy_count < min_non_legacy:
            needed = min_non_legacy - non_legacy_count
            print(f"\nStill need {needed} more non-legacy relations.")
            print(f"Trying Dixon's random squares method...")
            self._dixon_random_squares(needed)
        
        # Final count
        double_count = sum(1 for r in self.relations if 'type' in r and r['type'] == 'double')
        sieve_count = sum(1 for r in self.relations if 'd_exponents' in r)
        legacy_count = sum(1 for r in self.relations if 'exponents' in r and 'type' not in r)
        print(f"\nTotal relations: {len(self.relations)}")
        print(f"  Final breakdown: {double_count} double, {sieve_count} sieve, {legacy_count} legacy (will be skipped)")
        
        # Check if sieve relations have small x values (will lead to X == Y)
        if sieve_count > 0:
            sieve_x_values = [r['x'] for r in self.relations if 'd_exponents' in r]
            small_x_count = sum(1 for x in sieve_x_values if x * x < self.N)
            if small_x_count == sieve_count:
                print(f"  [WARNING] All {sieve_count} sieve relations have x² < N!")
                print(f"  This means x² mod N = x², which will lead to X == Y for all dependencies.")
                print(f"  SOLUTION: Need relations with larger x values (x² >= N).")
                print(f"  - Increase lattice dimension to find better relations")
                print(f"  - Or manually provide relations with larger x values")
            elif small_x_count > sieve_count // 2:
                print(f"  [WARNING] {small_x_count}/{sieve_count} sieve relations have x² < N.")
                print(f"  Many dependencies will be trivial (X == Y).")

    def _supplementary_sieve(self, needed):
        """Quick sieve to find additional relations if LLL didn't find enough."""
        print(f"  Starting supplementary sieve (target: {needed} relations)...")
        start_x = math.isqrt(self.N) + 1
        # Use a reasonable interval - not too large to avoid long waits
        interval_size = min(self.interval_size, 5000000)  # Reduced from 10M for faster execution
        
        prime_logs = [math.log(p) for p in self.primes]
        sieve_array = [0.0] * interval_size
        
        # Quick log sieve
        print(f"  Sieving interval [{start_x}, {start_x + interval_size}]...")
        for p_idx, p in enumerate(self.primes):
            log_p = prime_logs[p_idx]
            roots = self._tonelli_shanks(self.N, p)
            for r in roots:
                first_i = (r - start_x) % p
                for i in range(first_i, interval_size, p):
                    sieve_array[i] += log_p
        
        # Get top candidates - check more candidates if we need more relations
        threshold = max(100, needed * 200)  # Check more candidates
        indexed = [(sieve_array[i], i) for i in range(interval_size)]
        indexed.sort(reverse=True)
        
        found = 0
        smooth_found = 0
        for score, i in indexed[:threshold]:
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
                # Full relation (smooth) - add directly as sieve relation
                self.relations.append({
                    'x': x,
                    'd_exponents': d_exponents,
                    'remainder': 1
                })
                found += 1
                smooth_found += 1
                if smooth_found <= 5:
                    print(f"    Found smooth relation: x={x}, x²-N factors completely")
            elif jacobi_symbol(self.N, abs(temp)) == 1:
                # Partial relation - use Large Prime Variation
                lp = abs(temp)
                if lp in self.partial_relations:
                    # Match found! Add both
                    self.relations.append(self.partial_relations.pop(lp))
                    self.relations.append({
                        'x': x,
                        'd_exponents': d_exponents,
                        'remainder': lp
                    })
                    found += 2
                else:
                    # Store for later
                    self.partial_relations[lp] = {
                        'x': x,
                        'd_exponents': d_exponents,
                        'remainder': lp
                    }
            
            if found >= needed:
                break
        
        print(f"  Supplementary sieve found {found} additional relations ({smooth_found} smooth, {found - smooth_found} partial).")

    def _dixon_random_squares(self, needed):
        """
        Use Dixon's random squares method to find additional relations.
        Randomly select x and check if x² mod N is smooth over the factor base.
        Prefer x near sqrt(N) for better diversity.
        """
        print(f"  Looking for random Dixon squares (x² smooth mod N)...")

        found = 0
        attempts = 0
        max_attempts = 50000  # Limit attempts to avoid infinite loop

        sqrt_n = math.isqrt(self.N)
        n_bits = self.N.bit_length()
        
        # Determine appropriate x range based on N size
        if n_bits < 64:
            # Small N: can use smaller x values, but still avoid very small ones
            min_x = max(100, sqrt_n // 100)
            max_x = sqrt_n * 2
        elif n_bits < 256:
            # Medium N: use moderate x values
            min_x = max(1000, sqrt_n // 100)
            max_x = sqrt_n * 2
        else:
            # Large N: must use x near sqrt(N)
            min_x = max(10000, sqrt_n // 1000)
            max_x = sqrt_n + min(sqrt_n // 10, 1000000)

        while found < needed and attempts < max_attempts:
            attempts += 1
            
            # For small N, we can use a wider range
            # For large N, focus on x near sqrt(N)
            if n_bits < 128:
                # Small N: use wider range but avoid very small x
                if attempts % 5 == 1:
                    # Occasionally try smaller x (but not too small)
                    x = random.randint(min_x, sqrt_n // 2)
                else:
                    # Mostly use x in upper range
                    x = random.randint(sqrt_n // 2, max_x)
            else:
                # Large N: focus on x near sqrt(N)
                range_size = min(sqrt_n // 10, 1000000)
                x = random.randint(max(min_x, sqrt_n - range_size), min(max_x, sqrt_n + range_size))
            
            x_squared = (x * x) % self.N
            
            # For very small x, x² mod N = x², which gives similar relations
            # This leads to X == Y when combining relations
            # We need x² >= N for proper modular arithmetic
            x_squared_val = x * x
            if x_squared_val < self.N:
                # For small N, we might accept some small x, but limit them
                n_bits = self.N.bit_length()
                if n_bits < 32:
                    # Very small N: accept small x but prefer larger
                    if x < 10 and attempts % 10 != 1:  # Only 10% of small x
                        continue
                elif n_bits < 64:
                    # Small N: limit very small x
                    if x < 100 and attempts % 5 != 1:  # Only 20% of very small x
                        continue
                else:
                    # Larger N: skip all x where x² < N
                    continue
            
            # Check if x_squared is smooth
            if self._is_smooth(x_squared, self.primes):
                # Factor x_squared over the factor base
                exponents, remainder = self._factor_over_base(x_squared, self.primes)
                
                # Only accept if remainder is 1 (smooth) or if x is large enough
                if remainder == 1 or x * x >= self.N:
                    if found < 5:  # Only print first few
                        print(f"  Found smooth square: x={x}, x² ≡ {x_squared} mod N")
                    
                    self.relations.append({
                        'x': x,
                        'd_exponents': exponents,
                        'remainder': remainder
                    })
                    found += 1
                
                if found >= needed:
                    break
        
        print(f"  Found {found} random Dixon squares after {attempts} attempts.")
        return found > 0

    def solve_linear_system(self):
        print(f"\nSolving Linear System with {len(self.relations)} relations...")
        
        if len(self.relations) < 2:
            print("Too few relations to attempt solution.")
            return
            
        # Identify all extra factors (remainders)
        extra_factors = set()
        for rel in self.relations:
            r = rel.get('remainder', 1)
            if r != 1:
                extra_factors.add(r)
        
        sorted_extras = sorted(list(extra_factors))
        extra_map = {val: i for i, val in enumerate(sorted_extras)}
        
        d = len(self.primes)
        num_base_cols = d + 1 # -1 and primes
        
        # Matrix columns: [LHS_Base (d+1)] + [RHS_Base (d+1)] + [Extras]
        # Total width = 2*(d+1) + len(extras)
        
        base_block_size = num_base_cols
        total_cols = 2 * base_block_size + len(sorted_extras)
        
        print(f"  Matrix size: {len(self.relations)} x {total_cols} (LHS+RHS Base: {2*base_block_size}, Extra: {len(sorted_extras)})")
        
        # Count relation types for debugging
        double_count = sum(1 for r in self.relations if 'type' in r and r['type'] == 'double')
        sieve_count = sum(1 for r in self.relations if 'd_exponents' in r)
        legacy_count = sum(1 for r in self.relations if 'exponents' in r and 'type' not in r)
        print(f"  Relation types: {double_count} double, {sieve_count} sieve, {legacy_count} legacy (skipped)")
        
        # WARNING: If we only have legacy relations, we'll skip them all
        # because legacy relations have X = 1, so we're looking for Y^2 = 1 mod N (Y = ±1, trivial)
        if double_count == 0 and sieve_count == 0 and legacy_count > 0:
            print(f"  [WARNING] Only legacy relations found! These are being skipped because")
            print(f"  they represent 1 = RHS mod N, which always gives X=1 (trivial solutions).")
            print(f"  Need double or sieve relations for non-trivial factorizations.")
            print(f"  Consider increasing lattice dimension or improving relation finding.")

        M = []
        valid_indices = []
        
        # Skip legacy relations by default - they always lead to X=1 (trivial solutions)
        skip_legacy = True
        
        for idx, rel in enumerate(self.relations):
            row = [0] * total_cols
            
            if 'type' in rel and rel['type'] == 'double':
                # Double relation: [LHS_vec, RHS_vec]
                # vec length is 2 * base_block_size
                vec = rel['vec']
                if len(vec) != 2 * base_block_size:
                    print(f"Warning: Malformed double relation at index {idx}")
                    continue
                
                for i, e in enumerate(vec):
                    # Use e % 2 (not abs) since in GF(2), -1 = 1
                    # But we want to preserve the sign information correctly
                    row[i] = e % 2
                    
            elif 'd_exponents' in rel:
                # Sieve relation: x^2 = prod(p^e) * R
                # LHS is x^2 -> Exponents are all even (0 mod 2)
                # RHS is prod(p^e) * R
                
                # LHS columns (0 to d) are all 0
                
                # RHS columns (d+1 to 2d+1)
                # d_exponents maps to primes (indices 1 to d in the block)
                # Index 0 (-1) is 0 for squares
                
                rhs_offset = base_block_size
                
                # -1 exponent is 0
                row[rhs_offset + 0] = 0 
                
                for i, e in enumerate(rel['d_exponents']):
                    # Primes start at offset + 1
                    row[rhs_offset + 1 + i] = e % 2
                    
                # Extra factor
                r = rel.get('remainder', 1)
                if r != 1:
                    extra_idx = 2 * base_block_size + extra_map[r]
                    row[extra_idx] = 1
            
            elif 'exponents' in rel:
                 # Legacy/Fallback relation - SKIP BY DEFAULT
                 # These represent 1 = prod(p^e) * R, which always gives X=1 (trivial)
                 if skip_legacy:
                     continue  # Skip legacy relations
                 
                 # Legacy/Fallback relation
                 # Treated as 1 = prod(p^e) * R  =>  1^2 = prod(p^e) * R
                 # LHS = 1 (exponents 0)
                 # RHS = prod(p^e)
                 
                 rhs_offset = base_block_size
                 for i, e in enumerate(rel['exponents']):
                     # Use e % 2 for consistency (in GF(2), -1 = 1)
                     row[rhs_offset + i] = e % 2
                     
                 r = rel.get('remainder', 1)
                 if r != 1:
                    extra_idx = 2 * base_block_size + extra_map[r]
                    row[extra_idx] = 1
            else:
                continue
                
            M.append(row)
            valid_indices.append(idx)
        
        # Check if we have any valid relations after skipping legacy
        if len(M) == 0:
            print(f"  [ERROR] No valid relations to use (all were legacy type, which are skipped).")
            print(f"  Need to find 'double' or 'sieve' relations for factorization.")
            print(f"  Try:")
            print(f"    - Increasing lattice dimension")
            print(f"    - Running more LLL passes")
            print(f"    - Increasing factor base size")
            return
            
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
        
        # Verify kernel vectors are correct: they should be in the nullspace
        # Each kernel vector v should satisfy: M @ v = 0 (mod 2)
        print(f"  Found {len(kernel_basis)} independent dependencies (Kernel size).")
        
        # Validate kernel vectors (debug check)
        if len(kernel_basis) > 0 and len(kernel_basis[0]) == num_rows:
            # Check first few kernel vectors
            for idx, kb in enumerate(kernel_basis[:min(3, len(kernel_basis))]):
                # Compute M @ kb mod 2
                result = [0] * num_cols
                for j in range(num_rows):
                    if kb[j] == 1:
                        for c in range(num_cols):
                            result[c] = (result[c] + matrix[j][c]) % 2
                # Result should be all zeros
                if any(result):
                    print(f"  Warning: Kernel vector {idx} may be invalid (M @ v != 0)")
                elif idx == 0:
                    print(f"  Verified: Kernel vectors are in nullspace (checked first vector)")
        
        if not kernel_basis:
            print("  No dependencies found.")
            return

        # Randomized Search for Non-Trivial Factors
        import random
        
        # GEOMETRIC DIVINING:
        # The kernel is a high-dimensional space, but N = p*q means there's
        # a hidden 2D structure (mod p and mod q). 
        # We project the kernel onto 2D and search along the "factor directions".
        
        print(f"  Attempting to find non-trivial factors from {len(kernel_basis)} dependencies...")
        print(f"  Using geometric projection to divine factor directions...")
        
        # Project kernel basis to 2D using improved projection
        kernel_dim = len(kernel_basis)
        
        if kernel_dim <= 2:
            # For small kernels, just use the vectors directly
            kernel_2d = [(float(kb[0]) if len(kb) > 0 else 0, float(kb[1]) if len(kb) > 1 else 0) for kb in kernel_basis]
        else:
            # Use PCA-like projection: find directions of maximum variance
            # Compute covariance matrix of kernel vectors
            kernel_matrix = [[float(x) for x in kb] for kb in kernel_basis]
            
            # Find principal components (simplified PCA)
            # Compute mean
            means = [sum(row[i] for row in kernel_matrix) / len(kernel_matrix) for i in range(len(kernel_matrix[0]))]
            
            # Center the data
            centered = [[row[i] - means[i] for i in range(len(row))] for row in kernel_matrix]
            
            # Compute covariance matrix (simplified)
            cov = [[0] * len(means) for _ in range(len(means))]
            for row in centered:
                for i in range(len(row)):
                    for j in range(len(row)):
                        cov[i][j] += row[i] * row[j]
            for i in range(len(cov)):
                for j in range(len(cov)):
                    cov[i][j] /= len(centered)
            
            # Find eigenvector of largest eigenvalue (power iteration)
            v = [1.0] * len(means)
            for _ in range(10):  # 10 iterations
                v_new = [sum(cov[i][j] * v[j] for j in range(len(v))) for i in range(len(v))]
                norm = math.sqrt(sum(x*x for x in v_new))
                if norm > 0:
                    v = [x/norm for x in v_new]
            
            proj1 = v
            
            # Find second principal component (orthogonal to first)
            # Gram-Schmidt
            proj2 = [1.0] * len(means)
            dot = sum(proj2[i] * proj1[i] for i in range(len(proj1)))
            proj2 = [proj2[i] - dot * proj1[i] for i in range(len(proj1))]
            norm = math.sqrt(sum(x*x for x in proj2))
            if norm > 0:
                proj2 = [x/norm for x in proj2]
            
            # Project each kernel vector
            kernel_2d = []
            for kb in kernel_basis:
                x = sum(kb[i] * proj1[i] for i in range(len(kb)))
                y = sum(kb[i] * proj2[i] for i in range(len(kb)))
                kernel_2d.append((x, y))
        
        # We will try up to max_attempts total checks
        kernel_size = len(kernel_basis)
        if kernel_size <= 20:
            max_attempts = 5000
        elif kernel_size <= 100:
            max_attempts = 10000
        else:
            max_attempts = 20000  # More attempts for very large kernels
        trivial_count = 0
        
        # Pre-check: Count relation types to optimize dependency generation
        double_count = sum(1 for r in self.relations if 'type' in r and r['type'] == 'double')
        sieve_count = sum(1 for r in self.relations if 'd_exponents' in r)
        legacy_count = sum(1 for r in self.relations if 'exponents' in r and 'type' not in r)
        has_non_legacy = double_count > 0 or sieve_count > 0
        
        # Generator to yield masks (linear combinations of kernel basis)
        # Returns (mask, phase_name, debug_info)
        def dependency_generator():
            # Phase 1: Single basis vectors
            print("  Phase 1: Checking individual basis vectors...")
            for i in range(len(kernel_basis)):
                mask = [0] * len(kernel_basis)
                mask[i] = 1
                yield mask, "Phase1_Single", f"basis_vector_{i}"
            
            # Phase 1.5: Triangulation-based search (primary geometric method)
            if len(kernel_2d) >= 3:
                print("  Phase 1.5: Triangulation-based geometric search...")
                try:
                    points_2d = np.array(kernel_2d)
                    
                    # Ensure proper 2D distribution by checking variance
                    x_coords = points_2d[:, 0]
                    y_coords = points_2d[:, 1]
                    x_var = np.var(x_coords)
                    y_var = np.var(y_coords)
                    
                    # If degenerate, add small perturbations
                    if x_var < 1e-6 or y_var < 1e-6:
                        print("    Points are degenerate, adding perturbations...")
                        # Add small random noise to spread points
                        np.random.seed(42)
                        noise_x = np.random.normal(0, 0.01 * (np.max(np.abs(x_coords)) + 1e-6), len(points_2d))
                        noise_y = np.random.normal(0, 0.01 * (np.max(np.abs(y_coords)) + 1e-6), len(points_2d))
                        points_2d[:, 0] += noise_x
                        points_2d[:, 1] += noise_y
                    
                    # Use scipy's ConvexHull which is more robust than Delaunay for 2D
                    from scipy.spatial import ConvexHull
                    
                    # Compute convex hull
                    hull = ConvexHull(points_2d)
                    
                    # Find triangles within the convex hull
                    # Use Delaunay but with better error handling
                    from scipy.spatial import Delaunay
                    
                    # Try Delaunay with QJ option (joggle) to handle near-coplanar points
                    try:
                        tri = Delaunay(points_2d, qhull_options='QJ')
                    except:
                        # If still fails, use convex hull vertices only
                        hull_points = points_2d[hull.vertices]
                        if len(hull_points) >= 3:
                            tri = Delaunay(hull_points, qhull_options='QJ')
                        else:
                            raise ValueError("Insufficient hull points for triangulation")
                    
                    # Find triangles that are most promising
                    origin = np.array([0.0, 0.0])
                    triangle_candidates = []
                    
                    for simplex in tri.simplices:
                        triangle_points = points_2d[simplex]
                        
                        # Calculate triangle properties
                        centroid = np.mean(triangle_points, axis=0)
                        dist_to_origin = np.linalg.norm(centroid)
                        
                        # Check if triangle contains or is near origin
                        if self._point_in_triangle(origin, triangle_points):
                            priority = 0  # Highest priority
                        else:
                            # Distance from origin to triangle
                            dist = self._distance_to_triangle(origin, triangle_points)
                            priority = dist
                        
                        triangle_candidates.append((simplex, priority, dist_to_origin))
                    
                    # Sort by priority (closer to origin first)
                    triangle_candidates.sort(key=lambda x: x[1])
                    
                    # Generate combinations from best triangles
                    for simplex, priority, dist in triangle_candidates[:min(15, len(triangle_candidates))]:
                        # Try the full triangle
                        mask = [0] * len(kernel_basis)
                        for idx in simplex:
                            mask[idx] = 1
                        yield mask, "Phase1.5_Triangle", f"triangle_{simplex[0]}_{simplex[1]}_{simplex[2]}_priority_{priority:.3f}"
                        
                        # Try pairs from the triangle
                        for i in range(3):
                            for j in range(i+1, 3):
                                mask = [0] * len(kernel_basis)
                                mask[simplex[i]] = 1
                                mask[simplex[j]] = 1
                                yield mask, "Phase1.5_Triangle", f"pair_{simplex[i]}_{simplex[j]}_from_triangle"
                
                except Exception as e:
                    print(f"    Triangulation skipped: {e}")
                    print("    Continuing with other geometric methods...")
            
            # Phase 2: Adaptive Geometric search - focus on dense regions
            print("  Phase 2: Adaptive geometric divination along projected directions...")
            
            # Scale search parameters based on kernel size
            kernel_size = len(kernel_basis)
            if kernel_size <= 10:
                num_angles = 72
                max_combine = min(8, kernel_size)
                max_start_positions = 3
            elif kernel_size <= 50:
                num_angles = 48
                max_combine = min(6, kernel_size)
                max_start_positions = 2
            else:
                num_angles = 24  # Reduce for very large kernels
                max_combine = min(4, kernel_size)
                max_start_positions = 1
            
            # Find directions with high point density (adaptive angle selection)
            angles_to_try = []
            
            # Always include cardinal directions
            angles_to_try.extend([0, math.pi/2, math.pi, 3*math.pi/2])
            
            # Add directions toward clusters of points
            if len(kernel_2d) > 5:
                # Compute angles of all points from origin
                point_angles = []
                for x, y in kernel_2d:
                    if x != 0 or y != 0:
                        angle = math.atan2(y, x)
                        if angle < 0:
                            angle += 2 * math.pi
                        point_angles.append(angle)
                
                # Find dense regions (simple clustering)
                if point_angles:
                    point_angles.sort()
                    # Look for gaps larger than average
                    gaps = []
                    for i in range(len(point_angles)):
                        next_i = (i + 1) % len(point_angles)
                        gap = (point_angles[next_i] - point_angles[i]) % (2 * math.pi)
                        gaps.append((gap, point_angles[i]))
                    
                    gaps.sort(reverse=True)
                    # Add angles in the largest gaps (where clusters might be)
                    for gap_size, angle in gaps[:min(8, len(gaps)//2)]:
                        if gap_size > math.pi / 6:  # Only if gap is significant
                            cluster_angle = (angle + gap_size/2) % (2 * math.pi)
                            angles_to_try.append(cluster_angle)
            
            # Ensure we don't have too many angles
            angles_to_try = angles_to_try[:num_angles]
            
            for angle in angles_to_try:
                direction = (math.cos(angle), math.sin(angle))
                
                # Find kernel vectors that align with this direction
                scores = []
                for i, (x, y) in enumerate(kernel_2d):
                    # Dot product with direction (cosine similarity)
                    dot = x * direction[0] + y * direction[1]
                    # Also consider magnitude for ranking
                    magnitude = math.sqrt(x*x + y*y)
                    if magnitude > 0:
                        score = dot / magnitude  # Normalized dot product
                    else:
                        score = 0
                    scores.append((score, i))
                
                scores.sort(reverse=True)
                
                # Try various combination sizes, prioritizing higher alignment scores
                for num_combine in range(1, max_combine + 1):
                    # Try the top N for this combination size
                    for start_idx in range(min(max_start_positions, len(scores) - num_combine + 1)):
                        mask = [0] * len(kernel_basis)
                        selected_indices = []
                        for _, idx in scores[start_idx:start_idx + num_combine]:
                            mask[idx] = 1
                            selected_indices.append(idx)
                        yield mask, "Phase2_Geometric", f"angle_{int(angle*180/math.pi)}deg_top{num_combine}_start{start_idx}"
            
            # Phase 3: Random combinations (fallback)
            print("  Phase 3: Checking random combinations...")
            rand_counter = 0
            while True:
                rand_counter += 1
                mask = [random.randint(0, 1) for _ in range(len(kernel_basis))]
                if sum(mask) == 0: continue
                yield mask, "Phase3_Random", f"random_{rand_counter}"

        attempt_counter = 0
        current_phase = ""
        current_debug = ""
        
        for mask, phase, debug_info in dependency_generator():
            attempt_counter += 1
            attempt = attempt_counter
            current_phase = phase
            current_debug = debug_info
            if attempt_counter > max_attempts:
                break
                
            # Combine the dependency vectors
            combined_dependency = [0] * len(valid_indices)
            for k, bit in enumerate(mask):
                if bit:
                    for r in range(len(valid_indices)):
                        combined_dependency[r] ^= kernel_basis[k][r]
            
            indices = [k for k, bit in enumerate(combined_dependency) if bit == 1]
            if not indices: continue
            
            # Early triviality check: Skip if dependency selects all relations
            # (selecting all relations often leads to trivial solutions)
            if len(indices) == len(valid_indices):
                continue
            
            # Skip dependencies that select only 1 relation - these are often trivial
            # (a single relation by itself rarely gives non-trivial factorization)
            if len(indices) == 1:
                trivial_count += 1
                continue
            
            # Construct X and Y
            X_val = 1 # Accumulator for explicit X values (from sieve)
            
            # Exponent accumulators for base primes
            X_exponents = [0] * num_base_cols
            Y_exponents = [0] * num_base_cols
            
            Y_extra_factors = {}
            
            # Track relation types in this dependency
            has_double = False
            has_sieve = False
            has_legacy = False
            
            for idx in indices:
                real_idx = valid_indices[idx] # Map back to original relations
                rel = self.relations[real_idx]
                
                if 'type' in rel and rel['type'] == 'double':
                    has_double = True
                    vec = rel['vec']
                    # vec is [LHS..., RHS...]
                    for i in range(num_base_cols):
                        X_exponents[i] += vec[i]
                        Y_exponents[i] += vec[num_base_cols + i]
                        
                elif 'd_exponents' in rel:
                    has_sieve = True
                    # Sieve: x^2 = RHS
                    x_val = rel['x']
                    X_val = (X_val * x_val) % self.N
                    
                    # RHS exponents
                    for i, e in enumerate(rel['d_exponents']):
                        Y_exponents[i+1] += e
                    
                    # Extra
                    r = rel.get('remainder', 1)
                    if r != 1:
                        Y_extra_factors[r] = Y_extra_factors.get(r, 0) + 1
                        
                elif 'exponents' in rel:
                    has_legacy = True
                    # Legacy: These represent 1 = prod(p^e) * R
                    # Problem: X is always 1 for legacy relations, so we can only get trivial solutions
                    # Try to use the 'x' field if it exists (from large prime matching)
                    if 'x' in rel and rel['x'] != 1:
                        X_val = (X_val * rel['x']) % self.N
                    
                    for i, e in enumerate(rel['exponents']):
                        Y_exponents[i] += e
                    r = rel.get('remainder', 1)
                    if r != 1:
                        Y_extra_factors[r] = Y_extra_factors.get(r, 0) + 1
            
            # CRITICAL: If we only have legacy relations and X_val == 1, then X = 1
            # This means we're looking for Y such that Y^2 = 1 mod N, which gives Y = ±1 (trivial)
            # Skip these dependencies early - they can NEVER give non-trivial factors
            if not has_double and not has_sieve and X_val == 1 and all(e == 0 for e in X_exponents):
                trivial_count += 1
                # Skip immediately without expensive computation
                if attempt_counter % 1000 == 0 and attempt_counter <= 5000:
                    print(f"    Skipping legacy-only dependency (X=1, would give Y=±1)...")
                continue

            # CRITICAL: For sieve-only dependencies, check if all x values are small
            # If all x² < N, then x² mod N = x², and combining gives X == Y
            if has_sieve and not has_double:
                # Check if all x values in this dependency are small
                sieve_x_vals = [self.relations[valid_indices[idx]]['x'] 
                               for idx in indices 
                               if 'd_exponents' in self.relations[valid_indices[idx]]]
                if sieve_x_vals:
                    max_x_squared = max(x * x for x in sieve_x_vals)
                    if max_x_squared < self.N:
                        # All x values are small - this will give X == Y
                        trivial_count += 1
                        if attempt_counter <= 5:
                            print(f"    Attempt {attempt_counter}: Skipping (all x values small, x² < N, would give X==Y)")
                        continue
            
            # CRITICAL: If X_exponents == Y_exponents, then X and Y will have the same prime factors
            # This means X == Y (mod N) after taking square roots, which is trivial
            # The only exception is if X_val != 1 (from sieve relations), but even then it's likely trivial
            if X_exponents == Y_exponents:
                # If X_val == 1 and no extra factors, definitely trivial
                if X_val == 1 and len(Y_extra_factors) == 0:
                    trivial_count += 1
                    continue
                # Even with X_val != 1, if exponents match, X and Y will be very similar
                # Skip these early to save computation
                trivial_count += 1
                if attempt_counter <= 10:
                    print(f"    Attempt {attempt_counter}: Skipping (X_exponents == Y_exponents, likely X==Y)")
                continue
            
            # Check if X_exponents is all zeros (X == 1) or Y_exponents is all zeros (Y == 1)
            if all(e == 0 for e in X_exponents) and X_val == 1:
                trivial_count += 1
                continue
            if all(e == 0 for e in Y_exponents) and len(Y_extra_factors) == 0:
                trivial_count += 1
                continue
            
            # Calculate X and Y
            # X_val contains the product of x values from d_exponents relations
            # X_exponents contains exponents from 'double' type relations
            
            X = X_val
            Y = 1
            valid = True
            
            # 1. Base Primes for X (from 'double' relations only)
            # Handle -1 (index 0)
            if X_exponents[0] % 2 != 0: 
                valid = False
            else:
                # If sign bit is odd, multiply X by -1 (i.e., X = N - X)
                pass  # Actually handled by parity check
            
            for i in range(1, num_base_cols):
                exp = X_exponents[i]
                if exp % 2 != 0: 
                    valid = False
                    break
                if exp > 0:
                    X = (X * pow(self.primes[i-1], exp // 2, self.N)) % self.N
                
            if not valid: continue

            # 2. Base Primes for Y
            if Y_exponents[0] % 2 != 0: 
                valid = False
            
            if valid:
                for i in range(1, num_base_cols):
                    exp = Y_exponents[i]
                    if exp % 2 != 0: 
                        valid = False
                        break
                    if exp > 0:
                        Y = (Y * pow(self.primes[i-1], exp // 2, self.N)) % self.N
            
            if not valid: continue

            # 3. Extra Factors for Y
            for r, count in Y_extra_factors.items():
                if count % 2 != 0: 
                    valid = False
                    break
                Y = (Y * pow(r, count // 2, self.N)) % self.N
            
            if not valid: continue
            
            # Now X^2 = Y^2 mod N
            
            # CRITICAL VALIDATION: Verify that the dependency actually gives X^2 = Y^2 mod N
            # This should always be true if the matrix was constructed correctly,
            # but we verify it to catch any bugs in the linear algebra
            X2 = pow(X, 2, self.N)
            Y2 = pow(Y, 2, self.N)
            if X2 != Y2:
                # This should NEVER happen if the matrix is correct!
                print(f"    [ERROR] Matrix constraint violated! X^2 = {X2}, Y^2 = {Y2}")
                print(f"    This indicates a bug in matrix construction or dependency application.")
                print(f"    Dependency: {indices[:min(5, len(indices))]}... (showing first 5 relations)")
                continue  # Skip invalid relations
            
            # Additional triviality checks before expensive GCD
            # Check if X == 1 or Y == 1 (trivial square roots)
            if X == 1 or Y == 1:
                trivial_count += 1
                continue
            
            # Check if X == -1 mod N or Y == -1 mod N
            if X == (self.N - 1) % self.N or Y == (self.N - 1) % self.N:
                trivial_count += 1
                continue
            
            # Check if X != +/- Y
            # Note: If X^2 = 1, then Y^2 = 1.
            # If X is a non-trivial sqrt of 1 (i.e. X != 1 and X != -1), we found a factor!
            # So we ONLY skip if X == Y or X == -Y.
            
            if X == Y or X == (self.N - Y) % self.N:
                trivial_count += 1
                # Diagnostic: Track patterns in trivial solutions
                if attempt <= 10 or (attempt % 1000 == 0 and attempt <= 5000):
                    # Check if X_exponents == Y_exponents (would explain X == Y)
                    x_exp_str = str(X_exponents[:min(5, len(X_exponents))])
                    y_exp_str = str(Y_exponents[:min(5, len(Y_exponents))])
                    exp_match = (X_exponents == Y_exponents)
                    print(f"    Attempt {attempt}: Trivial (X={X}, Y={Y}, X_exp==Y_exp={exp_match})")
                    if attempt <= 5:
                        print(f"      X_exponents[:5]={x_exp_str}, Y_exponents[:5]={y_exp_str}")
                        print(f"      X_val={X_val}, has_double={has_double}, has_sieve={has_sieve}")
                continue
                
            # Check Factors: gcd(X - Y, N)
            val1 = (X - Y) % self.N
            val2 = (X + Y) % self.N
            f1 = math.gcd(val1, self.N)
            f2 = math.gcd(val2, self.N)
            
            if attempt < 5:
                print(f"    Debug: Non-trivial candidate! X={X}")
                print(f"    X^2 mod N = {X2}")
                print(f"    Y^2 mod N = {Y2}")
                print(f"    gcd(val1, N) = {f1}")
                print(f"    gcd(val2, N) = {f2}")
            
            if f1 > 1 and f1 < self.N:
                print(f"\n[SUCCESS] Factor found: {f1}")
                print(f"Other factor: {self.N // f1}")
                print(f"  >>> DIVINED FROM: {current_phase} ({current_debug})")
                print(f"  >>> Attempt #{attempt_counter}")
                return
            elif f2 > 1 and f2 < self.N:
                print(f"\n[SUCCESS] Factor found: {f2}")
                print(f"Other factor: {self.N // f2}")
                print(f"  >>> DIVINED FROM: {current_phase} ({current_debug})")
                print(f"  >>> Attempt #{attempt_counter}")
                return
        
        print(f"\n[FAILURE] Could not find non-trivial factors after {max_attempts} attempts.")
        print(f"Found {trivial_count} trivial solutions (all gave X == Y or X == -Y).")
        
        # Provide helpful diagnosis
        double_count = sum(1 for r in self.relations if 'type' in r and r['type'] == 'double')
        sieve_count = sum(1 for r in self.relations if 'd_exponents' in r)
        legacy_count = sum(1 for r in self.relations if 'exponents' in r and 'type' not in r)
        
        if double_count == 0 and sieve_count == 0:
            print(f"\n[DIAGNOSIS] All {legacy_count} relations are 'legacy' type (from large prime variation).")
            print(f"  Legacy relations have X = 1, so dependencies can only yield Y^2 = 1 mod N.")
            print(f"  This gives Y = ±1 mod N, which are trivial square roots.")
            print(f"  SOLUTION: Need to find more 'double' or 'sieve' relations.")
            print(f"  - Increase lattice dimension to find smoother relations")
            print(f"  - Run more LLL passes to find relations with remainder = ±1")
            print(f"  - Check if factor base size is appropriate")
        else:
            print(f"\n[DIAGNOSIS] Have {double_count} double + {sieve_count} sieve relations, but all dependencies are trivial.")
            print(f"  This suggests the relations might be too similar or the kernel structure is problematic.")
            print(f"  Possible issues:")
            print(f"  1. Relations might have X_exponents == Y_exponents (leading to X == Y)")
            print(f"  2. Kernel might only contain symmetric dependencies")
            print(f"  3. Need more diverse relations")
            print(f"  SOLUTIONS:")
            print(f"  - Try increasing lattice dimension for more diverse relations")
            print(f"  - Increase number of LLL passes")
            print(f"  - Check if relations are actually distinct")
            
            # Sample a few relations to see if they're diverse
            if len(self.relations) > 0:
                print(f"\n  Sampling first 3 relations:")
                for i, rel in enumerate(self.relations[:3]):
                    if 'type' in rel and rel['type'] == 'double':
                        vec = rel['vec']
                        lhs = vec[:len(vec)//2]
                        rhs = vec[len(vec)//2:]
                        print(f"    Relation {i}: LHS non-zero={sum(1 for x in lhs if x)}, RHS non-zero={sum(1 for x in rhs if x)}")
                    elif 'd_exponents' in rel:
                        print(f"    Relation {i}: Sieve, x={rel.get('x', '?')}, non-zero exponents={sum(1 for x in rel['d_exponents'] if x)}")

if __name__ == "__main__":
    # Target N provided by user
    # N = 2021
    # For 2048-bit testing (example)
    # N = ...
    
    # Let's use a generated 2048-bit number or the user's input
    # For now, we'll stick to the user's N=2021 for testing, but add logic for scaling
    
    # Example: 60-bit number
    # N = 115792089237316195423570985008687907853269984665640564039457584007913129639935
    
    # If user wants to test 2048-bit, they should set N here.
    # We will auto-scale lattice_dim based on N.
    
    N = 12
    
    print(f"Target N = {N} ({N.bit_length()} bits)")
    
    # Auto-scale lattice dimension
    lattice_dim = 100  # Force dimension 100 for testing
        
    print(f"Using lattice dimension: {lattice_dim}")
    
    factorizer = GeometricFactorizer(N, lattice_dim=lattice_dim)
    factorizer.find_relations()
    factorizer.solve_linear_system()
