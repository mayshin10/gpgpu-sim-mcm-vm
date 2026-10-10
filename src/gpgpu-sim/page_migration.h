// Page migration for CLAP (MICRO'25, Sec. 6 "Cooperation with Page Migrations").
//
// CLAP decides placement and page size only when a page is first mapped, so it
// cannot remap a data structure that a previous kernel already mapped. This
// module adds C-NUMA (Carrefour, ASPLOS'13) style co-location migration that is
// applied only to data structures shared across kernels:
//   - every L1 miss is sampled per 64KB unit and per requesting chiplet
//   - at each epoch, a page group (sized by CLAP's decision) that is accessed
//     dominantly by one remote chiplet is migrated to that chiplet
//   - the copy is injected into the inter-chip NoC as flits, so it consumes link
//     bandwidth and contends with regular remote traffic
//   - accesses to a group in flight are retried at the L1 TLB until the copy
//     completes; then the mapping is switched and the TLBs are shot down
//
// Built for compatibility with the GPGPU-Sim framework.

#ifndef PAGE_MIGRATION_H
#define PAGE_MIGRATION_H

#include <cstdint>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <vector>

class OptionParser;
class gpgpu_sim;
class memory_config;
class mem_fetch;
class Hub;
class tlb_tag_array;
class inter_noc;

class page_migration_config {
 public:
  void reg_options(OptionParser *opp);

  bool enable_page_migration;
  unsigned migration_epoch;         // decision period in core cycles
  unsigned migration_min_samples;   // minimum samples per group (Carrefour)
  float migration_dominance;        // share of the dominant chiplet
  float migration_remote_thres;     // per-malloc remote ratio gate (Carrefour LAR < 80%)
  unsigned migration_max_per_epoch; // max group migrations started per epoch
  unsigned migration_sample_rate;   // sample 1 out of N L1 misses
  unsigned migration_shootdown_lat; // fixed TLB shootdown latency in core cycles
};

struct migration_req {
  unsigned malloc_num;
  uint64_t base_vpn;  // 4KB vpn, aligned to n_pages
  unsigned n_pages;   // number of 4KB pages
  unsigned src;
  unsigned dst;
  uint64_t samples;
};

// interface of a migration policy
class page_migration_policy {
 public:
  virtual ~page_migration_policy() {}
  virtual void on_kernel_launch(unsigned kernel_seq) = 0;
  virtual void on_access(uint64_t vpn, unsigned chiplet, bool is_local) = 0;
  // called at the end of each epoch; epoch counters are reset afterwards
  virtual void decide(std::vector<migration_req> &out) = 0;
  virtual void print_stats(FILE *fout) const = 0;
};

class page_migration_engine;

// Carrefour page-local co-location restricted to cross-kernel shared mallocs
class cnuma_colocation_policy : public page_migration_policy {
 public:
  cnuma_colocation_policy(const page_migration_config &config, unsigned n_chiplet,
                          page_migration_engine *engine, Hub *hub, tlb_tag_array *shared_tlb);

  void on_kernel_launch(unsigned kernel_seq) override;
  void on_access(uint64_t vpn, unsigned chiplet, bool is_local) override;
  void decide(std::vector<migration_req> &out) override;
  void print_stats(FILE *fout) const override;

  unsigned group_pages(uint64_t vpn) const;

 private:
  struct unit_info {
    unsigned malloc_num;
    std::vector<uint64_t> count;  // per-chiplet samples in the current epoch
  };
  struct malloc_info {
    unsigned first_kernel_seq;
    uint64_t access;  // current epoch
    uint64_t remote;  // current epoch
  };

  bool malloc_eligible(unsigned malloc_num, const malloc_info &info) const;
  bool group_ready(uint64_t base_vpn, unsigned n_pages) const;

  const page_migration_config &m_config;
  unsigned m_n_chiplet;
  page_migration_engine *m_engine;
  Hub *m_hub;
  tlb_tag_array *m_shared_tlb;
  unsigned m_kernel_seq = 0;

  std::unordered_map<uint64_t, unit_info> m_units;  // key: 64KB unit (vpn >> 4)
  std::map<unsigned, malloc_info> m_mallocs;

  // skip reasons
  uint64_t m_skip_not_shared = 0;
  uint64_t m_skip_profiling = 0;
  uint64_t m_skip_local = 0;
  uint64_t m_skip_samples = 0;
  uint64_t m_skip_dominance = 0;
  uint64_t m_skip_already_local = 0;
  uint64_t m_skip_not_ready = 0;
  uint64_t m_skip_inflight = 0;
  uint64_t m_skip_limit = 0;
};

class page_migration_engine {
 public:
  page_migration_engine(const memory_config *mem_config, gpgpu_sim *gpu, Hub *hub,
                        tlb_tag_array *shared_tlb, inter_noc *noc);
  ~page_migration_engine();

  void cycle(uint64_t now);
  void on_access(const mem_fetch *mf);
  void on_kernel_launch(unsigned uid);
  void on_flit_arrival(uint64_t migration_id);
  bool is_frozen(uint64_t vpn) const;
  bool is_inflight(uint64_t base_vpn, unsigned n_pages) const;
  void print_stats(FILE *fout) const;

 private:
  struct inflight_migration {
    migration_req req;
    uint64_t remaining_flits;
    uint64_t start_cycle;
    uint64_t unfreeze_cycle;  // valid once all flits arrived
  };

  void start(const migration_req &req);
  void complete(uint64_t migration_id);
  void unfreeze(uint64_t migration_id);

  const memory_config *m_mem_config;
  const page_migration_config &m_config;
  gpgpu_sim *m_gpu;
  Hub *m_hub;
  tlb_tag_array *m_shared_tlb;
  inter_noc *m_noc;
  page_migration_policy *m_policy;

  unsigned m_sm_per_chiplet;
  uint64_t m_now = 0;
  uint64_t m_next_epoch = 0;
  uint64_t m_sample_cnt = 0;
  uint64_t m_next_id = 0;
  unsigned m_kernel_seq = 0;

  std::unordered_map<uint64_t, inflight_migration> m_inflight;  // key: migration id
  std::unordered_map<uint64_t, uint64_t> m_frozen;              // key: base vpn, value: migration id
  std::vector<uint64_t> m_wait_unfreeze;                        // shootdown in progress

  // stats
  uint64_t m_tot_started = 0;
  uint64_t m_tot_completed = 0;
  uint64_t m_tot_pages = 0;
  uint64_t m_tot_flits = 0;
  uint64_t m_tot_flits_arrived = 0;
  uint64_t m_tot_copy_cycles = 0;
  uint64_t m_tot_shootdowns = 0;
  std::map<unsigned, uint64_t> m_malloc_migrations;
  std::map<unsigned, uint64_t> m_size_migrations;  // key: group size in 4KB pages

  mutable uint64_t m_frozen_probe = 0;  // TLB probes retried due to an in-flight copy
};

#endif  // PAGE_MIGRATION_H
