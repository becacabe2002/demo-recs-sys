import numpy as np

class FixedPointExample:
    def __init__(self):
        # System parameters
        self.alpha = 16  # bits for rating fractional part
        self.beta = 8    # bits for learning parameters fractional part
        
        # Example ratings and parameters
        self.rating = 4.5
        self.gamma = 0.01    # learning rate
        self.lambda_reg = 0.02  # regularization for user profiles
        self.mu_reg = 0.02     # regularization for item profiles
        
    def to_fixed_point(self, number, bits):
        """Convert floating point to fixed point representation"""
        # Shift by specified bits to maintain precision
        scaled = int(number * (2 ** bits))
        return scaled
    
    def from_fixed_point(self, fixed_num, bits):
        """Convert fixed point back to floating point"""
        return fixed_num / (2 ** bits)
    
    def demonstrate_parameters(self):
        # Convert rating to fixed point
        rating_fixed = self.to_fixed_point(self.rating, self.alpha)
        
        # Convert learning parameters to fixed point
        gamma_fixed = self.to_fixed_point(self.gamma, self.beta)
        lambda_fixed = self.to_fixed_point(self.lambda_reg, self.beta)
        mu_fixed = self.to_fixed_point(self.mu_reg, self.beta)
        
        print("Rating Example (α=16 bits):")
        print(f"Original rating: {self.rating}")
        print(f"Fixed point: {rating_fixed}")
        print(f"Binary: {bin(rating_fixed)[2:].zfill(20)}")  # show 20 bits for clarity
        print(f"Recovered: {self.from_fixed_point(rating_fixed, self.alpha)}\n")
        
        print("Learning Parameters (β=8 bits):")
        print("Learning rate (γ):")
        print(f"Original: {self.gamma}")
        print(f"Fixed point: {gamma_fixed}")
        print(f"Binary: {bin(gamma_fixed)[2:].zfill(12)}")  # show 12 bits for clarity
        print(f"Recovered: {self.from_fixed_point(gamma_fixed, self.beta)}\n")
        
        print("Regularization Parameters (λ, μ):")
        print(f"Original λ: {self.lambda_reg}")
        print(f"Fixed point λ: {lambda_fixed}")
        print(f"Binary λ: {bin(lambda_fixed)[2:].zfill(12)}")
        print(f"Recovered λ: {self.from_fixed_point(lambda_fixed, self.beta)}\n")
        
        print(f"Original μ: {self.mu_reg}")
        print(f"Fixed point μ: {mu_fixed}")
        print(f"Binary μ: {bin(mu_fixed)[2:].zfill(12)}")
        print(f"Recovered μ: {self.from_fixed_point(mu_fixed, self.beta)}")

# Run demonstration
example = FixedPointExample()
example.demonstrate_parameters()
