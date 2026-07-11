from main import estimate_llama_dimensions

# Test for 150M parameters
hidden, layers, heads, kv_heads = estimate_llama_dimensions(150000000)
print("150M Model Dimensions:")
print(f"  hidden_size: {hidden}")
print(f"  num_hidden_layers: {layers}")
print(f"  num_attention_heads: {heads}")
print(f"  num_key_value_heads: {kv_heads}")
print(f"  GQA ratio: {heads}/{kv_heads} = {heads / kv_heads:.1f}")

# Calculate actual parameter count
# Llama params ≈ 12 * L * H^2
actual_params = 12 * layers * (hidden**2)
print(f"  Estimated params: {actual_params:,}")
print("  Target: 150,000,000")
print(f"  Difference: {abs(actual_params - 150000000):,}")
