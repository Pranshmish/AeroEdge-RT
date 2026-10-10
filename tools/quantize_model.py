import struct
import numpy as np
import os

def quantize_row(row):
    max_val = np.max(np.abs(row))
    if max_val < 1e-9:
        scale = 1.0
        q = np.zeros_like(row, dtype=np.int8)
    else:
        scale = float(max_val / 127.0)
        q = np.clip(np.round(row / scale), -127, 127).astype(np.int8)
    return q, scale

def quantize_matrix(mat):
    # mat shape: (rows, cols)
    rows, cols = mat.shape
    q_data = np.zeros((rows, cols), dtype=np.int8)
    scales = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        q_row, s = quantize_row(mat[i])
        q_data[i] = q_row
        scales[i] = s
    return q_data, scales

def build_quantized_model():
    model_path = "models/model.bin"
    with open(model_path, "rb") as f:
        cfg_bytes = f.read(28)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", cfg_bytes)
        print(f"Config: dim={dim}, hidden={hidden_dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}, seq_len={seq_len}")
        
        head_size = dim // n_heads
        kv_dim = (dim * n_kv_heads) // n_heads
        
        # Helper to read float32 array
        def read_floats(n):
            raw = f.read(n * 4)
            return np.frombuffer(raw, dtype=np.float32)
            
        token_emb = read_floats(vocab_size * dim).reshape(vocab_size, dim)
        rms_att = read_floats(n_layers * dim).reshape(n_layers, dim)
        wq = read_floats(n_layers * dim * dim).reshape(n_layers, dim, dim)
        wk = read_floats(n_layers * dim * kv_dim).reshape(n_layers, kv_dim, dim)
        wv = read_floats(n_layers * dim * kv_dim).reshape(n_layers, kv_dim, dim)
        wo = read_floats(n_layers * dim * dim).reshape(n_layers, dim, dim)
        rms_ffn = read_floats(n_layers * dim).reshape(n_layers, dim)
        w1 = read_floats(n_layers * hidden_dim * dim).reshape(n_layers, hidden_dim, dim)
        w2 = read_floats(n_layers * dim * hidden_dim).reshape(n_layers, dim, hidden_dim)
        w3 = read_floats(n_layers * hidden_dim * dim).reshape(n_layers, hidden_dim, dim)
        rms_final = read_floats(dim)
        freq_cis_real = read_floats(seq_len * (head_size // 2))
        freq_cis_imag = read_floats(seq_len * (head_size // 2))

    out_bin = "models/model_q8.bin"
    print(f"Quantizing layers and writing to {out_bin}...")
    
    with open(out_bin, "wb") as out:
        # 1. Config header (7 ints)
        out.write(cfg_bytes)
        
        # 2. Token embedding table (Float32 for precision in lookup)
        out.write(token_emb.tobytes())
        
        # 3. RMS Norms (Float32, small: ~few hundred bytes)
        out.write(rms_att.tobytes())
        out.write(rms_ffn.tobytes())
        out.write(rms_final.tobytes())
        
        # 4. RoPE freq_cis
        out.write(freq_cis_real.tobytes())
        out.write(freq_cis_imag.tobytes())
        
        # 5. Layer weights (wq, wk, wv, wo, w1, w2, w3) for all layers
        # For each matrix: write data bytes (rows * cols), then scales (rows * float32)
        for l in range(n_layers):
            matrices = [
                ("wq", wq[l]),
                ("wk", wk[l]),
                ("wv", wv[l]),
                ("wo", wo[l]),
                ("w1", w1[l]),
                ("w2", w2[l]),
                ("w3", w3[l])
            ]
            for name, mat in matrices:
                q_data, scales = quantize_matrix(mat)
                out.write(q_data.tobytes())
                out.write(scales.tobytes())
                
        # 6. Classifier head (wcls = token_emb)
        q_cls, s_cls = quantize_matrix(token_emb)
        out.write(q_cls.tobytes())
        out.write(s_cls.tobytes())

    size_orig = os.path.getsize(model_path)
    size_q8 = os.path.getsize(out_bin)
    print(f"Quantization complete! Original: {size_orig} bytes -> Quantized Q8: {size_q8} bytes ({size_q8/size_orig*100:.1f}%)")

if __name__ == "__main__":
    build_quantized_model()
