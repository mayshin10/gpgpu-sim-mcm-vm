// Page migration for CLAP (MICRO'25, Sec. 6 "Cooperation with Page Migrations").
// See page_migration.h for the overview.
//
// Built for compatibility with the GPGPU-Sim framework.

#include "page_migration.h"

#include <algorithm>
#include <cassert>

#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../option_parser.h"
#include "gpu-sim.h"
#include "inter_chip_interconnect.h"
#include "mem_fetch.h"
#include "tlb.h"

static const uint64_t VA_MASK_48 = UINT64_MAX >> 16;  // same parsing mask as Hub::translate

/******************************************************************************/
void page_migration_config::reg_options(OptionParser *opp) {
  option_parser_register(opp, "-enable_page_migration", OPT_BOOL, &enable_page_migration,
                         "Enable page migration for cross-kernel shared data (CLAP + migration)", "0");
  option_parser_register(opp, "-migration_epoch", OPT_UINT32, &migration_epoch,
                         "Page migration decision period in core cycles", "10000");
  option_parser_register(opp, "-migration_min_samples", OPT_UINT32, &migration_min_samples,
                         "Minimum number of samples for a page group to be migrated", "2");
  option_parser_register(opp, "-migration_dominance", OPT_FLOAT, &migration_dominance,
                         "Minimum access share of the dominant chiplet for co-location", "0.9");
  option_parser_register(opp, "-migration_remote_thres", OPT_FLOAT, &migration_remote_thres,
                         "Minimum remote access ratio of a data structure in an epoch", "0.2");
  option_parser_register(opp, "-migration_max_per_epoch", OPT_UINT32, &migration_max_per_epoch,
                         "Maximum number of page group migrations started per epoch", "64");
  option_parser_register(opp, "-migration_sample_rate", OPT_UINT32, &migration_sample_rate,
                         "Sample 1 out of N L1 misses (1 = all)", "1");
  option_parser_register(opp, "-migration_shootdown_lat", OPT_UINT32, &migration_shootdown_lat,
                         "Fixed TLB shootdown latency in core cycles", "0");
}

/******************************************************************************/
cnuma_colocation_policy::cnuma_colocation_policy(const page_migration_config &config, unsigned n_chiplet,
                                                 page_migration_engine *engine, Hub *hub,
                                                 tlb_tag_array *shared_tlb)
    : m_config(config), m_n_chiplet(n_chiplet), m_engine(engine), m_hub(hub), m_shared_tlb(shared_tlb) {}

void cnuma_colocation_policy::on_kernel_launch(unsigned kernel_seq) {
  m_kernel_seq = kernel_seq;
  // samples of the previous kernel do not describe the new access pattern
  for (auto &unit : m_units) {
    std::fill(unit.second.count.begin(), unit.second.count.end(), 0);
  }
  for (auto &m : m_mallocs) {
    m.second.access = 0;
    m.second.remote = 0;
  }
}

void cnuma_colocation_policy::on_access(uint64_t vpn, unsigned chiplet, bool is_local) {
  const uint64_t unit_key = vpn >> static_cast<uint64_t>(PAGE_64K_SHIFT);
  auto it = m_units.find(unit_key);
  if (it == m_units.end()) {
    unsigned malloc_num = m_hub->get_malloc(vpn << static_cast<uint64_t>(PAGE_OFFSET));
    unit_info info;
    info.malloc_num = malloc_num;
    info.count.assign(m_n_chiplet, 0);
    it = m_units.emplace(unit_key, info).first;
    if (malloc_num != static_cast<unsigned>(-1) && m_mallocs.find(malloc_num) == m_mallocs.end()) {
      m_mallocs[malloc_num] = malloc_info{m_kernel_seq, 0, 0};
    }
  }
  it->second.count[chiplet]++;

  const unsigned malloc_num = it->second.malloc_num;
  if (malloc_num != static_cast<unsigned>(-1)) {
    malloc_info &info = m_mallocs.at(malloc_num);
    info.access++;
    if (!is_local) info.remote++;
  }
}

// group size in 4KB pages: the unit CLAP mapped contiguously
unsigned cnuma_colocation_policy::group_pages(uint64_t vpn) const {
  if (m_shared_tlb->get_page_size(vpn) == PAGE_2M) return PAGE_2M;  // promoted 2MB page

  const unsigned block_status = m_hub->get_block_status(vpn >> static_cast<uint64_t>(PAGE_2M_SHIFT));
  if (block_status == MAPPED_AFTER_DECISION) {
    unsigned malloc_num = m_hub->get_malloc(vpn << static_cast<uint64_t>(PAGE_OFFSET));
    unsigned decision = m_hub->get_malloc_decision(malloc_num);
    if (decision >= PAGE_64K) return decision;
  }
  return PAGE_64K;
}

bool cnuma_colocation_policy::malloc_eligible(unsigned malloc_num, const malloc_info &info) const {
  return info.first_kernel_seq < m_kernel_seq &&
         m_hub->get_malloc_decision(malloc_num) != 0 &&
         static_cast<float>(info.remote) >= m_config.migration_remote_thres * static_cast<float>(info.access);
}

// every page of the group is mapped and no 2MB reservation is pending for it
bool cnuma_colocation_policy::group_ready(uint64_t base_vpn, unsigned n_pages) const {
  uint64_t check_base = base_vpn;
  uint64_t check_pages = n_pages;
  const unsigned block_status = m_hub->get_block_status(base_vpn >> static_cast<uint64_t>(PAGE_2M_SHIFT));
  if (block_status == MAPPED_BEFORE_DECISION && n_pages < PAGE_2M) {
    // opportunistic 2MB reservation may still be pending for the rest of the block
    check_base = (base_vpn >> static_cast<uint64_t>(PAGE_2M_SHIFT)) << static_cast<uint64_t>(PAGE_2M_SHIFT);
    check_pages = PAGE_2M;
  }
  for (uint64_t vpn = check_base; vpn < check_base + check_pages; vpn++) {
    void *pa = m_hub->translate(reinterpret_cast<void *>(vpn << static_cast<uint64_t>(PAGE_OFFSET)));
    if (reinterpret_cast<uint64_t>(pa) == UINT64_MAX) return false;
  }
  return true;
}

void cnuma_colocation_policy::decide(std::vector<migration_req> &out) {
  // data structure level gate
  std::map<unsigned, bool> eligible;
  for (auto &m : m_mallocs) {
    const malloc_info &info = m.second;
    if (info.access == 0) continue;
    bool ok = malloc_eligible(m.first, info);
    if (!ok) {
      if (info.first_kernel_seq >= m_kernel_seq) m_skip_not_shared++;
      else if (m_hub->get_malloc_decision(m.first) == 0) m_skip_profiling++;
      else m_skip_local++;
    }
    eligible[m.first] = ok;
  }

  // aggregate 64KB units into page groups
  struct group_acc {
    unsigned malloc_num;
    unsigned n_pages;
    std::vector<uint64_t> count;
  };
  std::map<uint64_t, group_acc> groups;
  for (auto &unit : m_units) {
    const unit_info &info = unit.second;
    if (info.malloc_num == static_cast<unsigned>(-1)) continue;
    auto e = eligible.find(info.malloc_num);
    if (e == eligible.end() || !e->second) continue;

    uint64_t total = 0;
    for (uint64_t c : info.count) total += c;
    if (total == 0) continue;

    const uint64_t unit_vpn = unit.first << static_cast<uint64_t>(PAGE_64K_SHIFT);
    const unsigned n_pages = group_pages(unit_vpn);
    const uint64_t base_vpn = unit_vpn & ~(static_cast<uint64_t>(n_pages) - 1);
    auto g = groups.find(base_vpn);
    if (g == groups.end()) {
      g = groups.emplace(base_vpn, group_acc{info.malloc_num, n_pages, std::vector<uint64_t>(m_n_chiplet, 0)}).first;
    }
    for (unsigned c = 0; c < m_n_chiplet; c++) g->second.count[c] += info.count[c];
  }

  // page-local co-location decision
  std::vector<migration_req> candidates;
  for (auto &g : groups) {
    const group_acc &acc = g.second;
    uint64_t total = 0;
    unsigned dst = 0;
    for (unsigned c = 0; c < m_n_chiplet; c++) {
      total += acc.count[c];
      if (acc.count[c] > acc.count[dst]) dst = c;
    }
    if (total < m_config.migration_min_samples) { m_skip_samples++; continue; }
    if (static_cast<float>(acc.count[dst]) < m_config.migration_dominance * static_cast<float>(total)) {
      m_skip_dominance++;
      continue;
    }
    const unsigned src = m_hub->get_map_chiplet(g.first);
    if (src == dst) { m_skip_already_local++; continue; }
    if (m_engine->is_inflight(g.first, acc.n_pages)) { m_skip_inflight++; continue; }
    if (!group_ready(g.first, acc.n_pages)) { m_skip_not_ready++; continue; }
    candidates.push_back(migration_req{acc.malloc_num, g.first, acc.n_pages, src, dst, total});
  }

  // hottest groups first
  std::sort(candidates.begin(), candidates.end(),
            [](const migration_req &a, const migration_req &b) { return a.samples > b.samples; });
  if (candidates.size() > m_config.migration_max_per_epoch) {
    m_skip_limit += candidates.size() - m_config.migration_max_per_epoch;
    candidates.resize(m_config.migration_max_per_epoch);
  }
  out.insert(out.end(), candidates.begin(), candidates.end());

  // reset epoch counters (Carrefour resets its statistics every period)
  for (auto &unit : m_units) {
    std::fill(unit.second.count.begin(), unit.second.count.end(), 0);
  }
  for (auto &m : m_mallocs) {
    m.second.access = 0;
    m.second.remote = 0;
  }
}

void cnuma_colocation_policy::print_stats(FILE *fout) const {
  fprintf(fout, "page_migration_skip_not_shared = %lu\n", m_skip_not_shared);
  fprintf(fout, "page_migration_skip_profiling = %lu\n", m_skip_profiling);
  fprintf(fout, "page_migration_skip_low_remote = %lu\n", m_skip_local);
  fprintf(fout, "page_migration_skip_samples = %lu\n", m_skip_samples);
  fprintf(fout, "page_migration_skip_dominance = %lu\n", m_skip_dominance);
  fprintf(fout, "page_migration_skip_already_local = %lu\n", m_skip_already_local);
  fprintf(fout, "page_migration_skip_inflight = %lu\n", m_skip_inflight);
  fprintf(fout, "page_migration_skip_not_ready = %lu\n", m_skip_not_ready);
  fprintf(fout, "page_migration_skip_limit = %lu\n", m_skip_limit);
}

/******************************************************************************/
page_migration_engine::page_migration_engine(const memory_config *mem_config, gpgpu_sim *gpu, Hub *hub,
                                             tlb_tag_array *shared_tlb, inter_noc *noc)
    : m_mem_config(mem_config), m_config(mem_config->m_migration_config), m_gpu(gpu), m_hub(hub),
      m_shared_tlb(shared_tlb), m_noc(noc) {
  assert(m_mem_config->enable_profiling && "page migration is implemented on top of CLAP (-enable_profiling 1)");
  assert(m_mem_config->mapping_granularity == PAGE_64K);
  assert(m_config.migration_epoch > 0);
  assert(m_config.migration_sample_rate > 0);
  m_sm_per_chiplet = m_mem_config->total_sm_num / m_mem_config->chiplet_num;
  m_policy = new cnuma_colocation_policy(m_config, m_mem_config->chiplet_num, this, hub, shared_tlb);
}

page_migration_engine::~page_migration_engine() { delete m_policy; }

void page_migration_engine::cycle(uint64_t now) {
  m_now = now;

  // finish shootdowns
  for (auto it = m_wait_unfreeze.begin(); it != m_wait_unfreeze.end();) {
    if (m_inflight.at(*it).unfreeze_cycle <= now) {
      unfreeze(*it);
      it = m_wait_unfreeze.erase(it);
    } else {
      ++it;
    }
  }

  if (now < m_next_epoch) return;
  m_next_epoch = now + m_config.migration_epoch;

  std::vector<migration_req> reqs;
  m_policy->decide(reqs);
  for (const migration_req &req : reqs) start(req);
}

void page_migration_engine::on_access(const mem_fetch *mf) {
  if (m_config.migration_sample_rate > 1 && (m_sample_cnt++ % m_config.migration_sample_rate) != 0) return;
  if (mf->get_sid() == static_cast<unsigned>(-1)) return;
  const uint64_t vpn = (mf->get_original_addr() & VA_MASK_48) >> static_cast<uint64_t>(PAGE_OFFSET);
  m_policy->on_access(vpn, mf->compute_origin_chiplet(), mf->get_is_local_access());
}

void page_migration_engine::on_kernel_launch(unsigned uid) {
  m_policy->on_kernel_launch(++m_kernel_seq);
  m_next_epoch = m_now + m_config.migration_epoch;
}

void page_migration_engine::start(const migration_req &req) {
  const uint64_t id = m_next_id++;
  const uint64_t bytes = static_cast<uint64_t>(req.n_pages) << static_cast<uint64_t>(PAGE_OFFSET);
  const uint64_t n_flits = bytes / static_cast<uint64_t>(FLIT_BYTES);
  assert(req.src != req.dst);

  m_inflight[id] = inflight_migration{req, n_flits, m_now, 0};
  m_frozen[req.base_vpn] = id;

  // copy traffic shares the inter-chip links with regular remote accesses
  for (uint64_t i = 0; i < n_flits; i++) {
    inter_flit *flit = new inter_flit(req.src, req.dst, nullptr, false);
    flit->m_is_migration = true;
    flit->m_migration_id = id;
    m_noc->push_flit(flit);
  }

  m_tot_started++;
  m_tot_flits += n_flits;
}

void page_migration_engine::on_flit_arrival(uint64_t migration_id) {
  auto it = m_inflight.find(migration_id);
  assert(it != m_inflight.end());
  assert(it->second.remaining_flits > 0);
  m_tot_flits_arrived++;
  if (--it->second.remaining_flits == 0) complete(migration_id);
}

void page_migration_engine::complete(uint64_t migration_id) {
  inflight_migration &mig = m_inflight.at(migration_id);
  const migration_req &req = mig.req;
  const uint64_t now = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;

  bool done = m_hub->migrate_group(req.base_vpn, req.n_pages, req.dst);
  assert(done && "page group became unmapped during migration");
  assert(m_hub->get_map_chiplet(req.base_vpn) == req.dst);

  // a group never crosses a 2MB chunk, so one chunk flush covers it
  m_shared_tlb->flush_TLBs(req.base_vpn >> static_cast<uint64_t>(PAGE_2M_SHIFT));

  m_tot_completed++;
  m_tot_shootdowns++;
  m_tot_pages += req.n_pages;
  m_tot_copy_cycles += now - mig.start_cycle;
  m_malloc_migrations[req.malloc_num]++;
  m_size_migrations[req.n_pages]++;

  if (m_config.migration_shootdown_lat == 0) {
    unfreeze(migration_id);
  } else {
    mig.unfreeze_cycle = now + m_config.migration_shootdown_lat;
    m_wait_unfreeze.push_back(migration_id);
  }
}

void page_migration_engine::unfreeze(uint64_t migration_id) {
  m_frozen.erase(m_inflight.at(migration_id).req.base_vpn);
  m_inflight.erase(migration_id);
}

bool page_migration_engine::is_frozen(uint64_t vpn) const {
  if (m_frozen.empty()) return false;
  for (uint64_t n_pages = PAGE_64K; n_pages <= PAGE_2M; n_pages <<= 1) {
    auto it = m_frozen.find(vpn & ~(n_pages - 1));
    if (it != m_frozen.end() && m_inflight.at(it->second).req.n_pages == n_pages) {
      m_frozen_probe++;
      return true;
    }
  }
  return false;
}

bool page_migration_engine::is_inflight(uint64_t base_vpn, unsigned n_pages) const {
  for (auto &f : m_frozen) {
    const uint64_t f_base = f.first;
    const uint64_t f_pages = m_inflight.at(f.second).req.n_pages;
    if (f_base < base_vpn + n_pages && base_vpn < f_base + f_pages) return true;
  }
  return false;
}

void page_migration_engine::print_stats(FILE *fout) const {
  fprintf(fout, "\n----------------------------Page Migration----------------------------\n");
  fprintf(fout, "page_migration_started = %lu\n", m_tot_started);
  fprintf(fout, "page_migration_completed = %lu\n", m_tot_completed);
  fprintf(fout, "page_migration_inflight = %lu\n", static_cast<uint64_t>(m_inflight.size()));
  fprintf(fout, "page_migration_pages_4KB = %lu\n", m_tot_pages);
  fprintf(fout, "page_migration_bytes = %lu\n", m_tot_pages << static_cast<uint64_t>(PAGE_OFFSET));
  fprintf(fout, "page_migration_flits = %lu\n", m_tot_flits);
  fprintf(fout, "page_migration_flits_arrived = %lu\n", m_tot_flits_arrived);
  fprintf(fout, "page_migration_avg_copy_cycles = %.2f\n",
          m_tot_completed == 0 ? 0.0 : static_cast<double>(m_tot_copy_cycles) / static_cast<double>(m_tot_completed));
  fprintf(fout, "page_migration_shootdowns = %lu\n", m_tot_shootdowns);
  fprintf(fout, "page_migration_frozen_retries = %lu\n", m_frozen_probe);
  for (auto &m : m_malloc_migrations) {
    fprintf(fout, "page_migration_malloc[%u] = %lu\n", m.first, m.second);
  }
  for (auto &s : m_size_migrations) {
    fprintf(fout, "page_migration_group_size[%uKB] = %lu\n", s.first * 4, s.second);
  }
  m_policy->print_stats(fout);
}
