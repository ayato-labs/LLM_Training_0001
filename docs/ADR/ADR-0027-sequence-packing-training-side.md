# ADR-0027-sequence-packing-training-side: Dynamic Sequence Packing on Training Orchestrator Side

- **Status:** Accepted
- **Date:** 2026-07-16
- **Deciders:** Solo Developer

## Context

Sequence packing (concatenating tokenized texts with `<eos>` and chunking them into fixed sizes of `seq_len` to eliminate pad tokens) is essential for maximizing GPU throughput during scratch pre-training.

When implementing this, we have two primary architectural choices:
1. **Offline Preprocessing (`DataPreprocessing`):** Apply sequence packing and chunking offline, storing the fully packed, ready-to-train datasets in files.
2. **On-the-Fly / Training Orchestrator (`LLM_Training`):** Keep the raw tokenized documents intact, and dynamically pack/chunk them at the beginning of the training process.

We must decide the placement of sequence packing to maintain the project's goal of flexible configuration, storage efficiency, and alignment with modern LLM training standards.

## Decision

We will implement sequence packing (`PackedDatasetWrapper`) on the **training orchestrator side** (`LLM_Training` under `src/training/model_utils.py` and `src/training/train_engine.py`) rather than in the offline preprocessing pipeline (`DataPreprocessing`).

### Rationale

1. **Context Length (`seq_len`) Flexibility**: 
   Sequence packing depends directly on the target model's context length. If sequence packing were implemented offline, experimenting with different context window lengths (e.g., 512, 1024, 2048, or 4096) would require rebuilding and storing separate massive datasets for each experiment. By packing on-the-fly, we can change `seq_len` dynamically in `config.yaml` and the training pipeline will immediately adapt.
   
2. **Storage and Versioning Optimization**:
   Pre-packed datasets lock files into specific sequences. By storing only the raw tokenized texts, we avoid storing multiple packed variations on disk, keeping our storage requirements low and clean.
   
3. **Framework Standard Practice**:
   Modern LLM frameworks (such as Hugging Face TRL's `ConstantLengthDataset` or standard PyTorch collators) apply packing dynamically at the training loader or dataset initialization step. Our implementation of `PackedDatasetWrapper` conforms to these industry conventions.

## Consequences

### Pros
- **Flexibility**: We can change context size (`seq_len`) dynamically in `config.yaml` without rebuilding data.
- **Storage Conservation**: No redundant pre-packed dataset copies are saved on local disk.
- **Ease of Verification**: Ensures that train and validation dataset splits are formatted identically using the same dynamic logic at runtime.

### Cons
- Slight start-up CPU delay (several seconds) at the beginning of training due to dynamic sequence packing. This is minor compared to hours of GPU training time.
