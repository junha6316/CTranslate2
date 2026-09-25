#pragma once

#include "storage_view.h"

namespace ctranslate2 {

  // Caller-owned device buffers reused across sampling calls: without them the GPU
  // branch allocates and frees two fresh device tensors on every decoding step, and the
  // CUDA TopK allocates and frees its two temporary reduction buffers. The TopK scratch
  // size is decode-constant (batch x k x blocks-per-beam), so those two reserve once.
  struct SamplerStaging {
    StorageView ids;
    StorageView scores;
    StorageView topk_tmp_ids;
    StorageView topk_tmp_vals;
  };

  // Base class for sampling from a score distribution.
  class Sampler {
  public:
    virtual ~Sampler() = default;

    // sample_ids and sampled_scores should be on CPU device.
    void operator()(const StorageView& scores,
                    StorageView& sampled_ids,
                    StorageView& sampled_scores,
                    dim_t num_samples = 1,
                    SamplerStaging* staging = nullptr) const;
  protected:
    virtual void sample(const StorageView& scores,
                        dim_t num_samples,
                        StorageView& sampled_ids,
                        StorageView& sampled_scores,
                        SamplerStaging* staging) const = 0;
  };


  class BestSampler : public Sampler {
  protected:
    void sample(const StorageView& scores,
                dim_t num_samples,
                StorageView& sampled_ids,
                StorageView& sampled_scores,
                SamplerStaging* staging) const final;
  };


  class RandomSampler : public Sampler {
  public:
    RandomSampler(dim_t from_topk = 0, float topp = 1, float temperature = 1);
  protected:
    void sample(const StorageView& scores,
                dim_t num_samples,
                StorageView& sampled_ids,
                StorageView& sampled_scores,
                SamplerStaging* staging) const final;
  private:
    dim_t _from_topk;
    float _topp;
    float _temperature;
  };

}
