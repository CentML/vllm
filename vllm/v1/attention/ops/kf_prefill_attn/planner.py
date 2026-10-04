# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Exact C++ port of the host planner (``setup()``) of the Kernel Factory solution in
``vllm/third_party/kf_prefill_attn/kernel.py`` (KF candidate 511e91b9; bit-identical
to it on 6,856 launches, see X/analysis/kf-prefill-attn). Re-port (or re-verify)
whenever kernel.py is swapped.

build() JIT-builds the CPU extension once per process (torch.utils.cpp_extension.load_inline,
-O3, no fp contraction) under VLLM_CACHE_ROOT/kf_prefill_attn_planner (keyed by the source
digest; $KF_PLANNER_BUILD_DIR overrides).  plan_into() runs the whole planning part of setup() --
_resid_scale, both _build_plan rounds (incl. the _SCOREMODE re-search), the
flat/pair pick and the variant logic -- with the GIL released, and writes what
setup() puts into st["plan"] (incl. the column-11 rewrite), st["bins"],
st["mrg"] and st["mbin"] into caller-owned int32 CPU tensors.

The search space is configurable: kgrid / base_waves / dense_waves / filler
mirror the module globals _KGRID / _BASE_WAVE_GRID / _DENSE_WAVE_GRID and
_filler_splits (filler=False == kfpatch's no-op _filler_splits).  The defaults
are the KF module's values (preset R0 = exact KF); PRESETS holds R0..R3 as in
kfpatch.py.

Return value of plan_into:
  (nwork, nbins_plus1, nmrg, nmbin_plus1, nslot, ngrp, variant, precise_variant, grid, use_mcast)
  nwork  = rows written to plan_out (= setup's max(len(plan), 1))
  nbins_plus1 = len(st["bins"]); nmrg = st["mrg"].shape[0] (= max(len(mrg), 1), a single
  zero row when there are no merge tasks); nmbin_plus1 = len(st["mbin"]) (= grid + 1)
  counter tensor length = max(4 * ngrp, 4) + 4
  Sentinels in nwork (outputs untouched unless noted):
    -1  an output capacity is too small; nbins_plus1/nmrg/nmbin_plus1 and the scalars are still
        returned (plan rows needed <= 2 * sum(ceil(q / 16)) + NSLOT_CAP (1520))
    -2  no work items (every query length is 0) -- setup() raises TypeError there
    -3  invalid input (a query length < 0, seq_len < query length, nsm < 1, or a merge
        layout with fewer than NMRG grid CTAs, where setup() raises IndexError)
"""

import hashlib
import os
from typing import Any

import torch

KGRID = tuple(
    sorted(
        set(
            (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
            + tuple(2.0 ** (i / 8.0) for i in range(41))
        )
    )
)
BASE_WAVE_GRID = (0.0625, 0.125, 0.25, 1.0 / 3.0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16)
DENSE_WAVE_GRID = tuple(
    sorted(
        set(
            BASE_WAVE_GRID
            + tuple(i / 16.0 for i in range(1, 17))
            + tuple(i / 4.0 for i in range(5, 33))
            + tuple(i / 2.0 for i in range(17, 33))
        )
    )
)

# name -> (kgrid, base_waves, dense_waves, filler); mirrors kfpatch.py
PRESETS = {
    "R0": (KGRID, BASE_WAVE_GRID, DENSE_WAVE_GRID, True),
    "R1": ((1.0, 2.0, 4.0, 8.0), BASE_WAVE_GRID, (), False),
    "R2": ((), BASE_WAVE_GRID, (), False),
    "R3": ((), (0.25, 0.5, 1, 2, 4), (), False),
}

PLAN_F = 12
MRG_F = 8

_CPP = r"""
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int64_t PAGE = 128;
constexpr int64_t HKV = 2;
constexpr int64_t NMRG = 4;
constexpr int64_t PLAN_F = 12;
constexpr int64_t MRG_F = 8;
constexpr int64_t NOPS = 60;
constexpr int64_t NSLOT_CAP = 1520;
constexpr double C_ITEM = 2.5;
constexpr double C_SPLIT = 0.5;
constexpr double C_MERGE = 5.0;
constexpr double C_MPRE = 0.2;
constexpr double C_MTASK = 10.0;
constexpr double RESID_ALPHA = 0.36;
constexpr int RESID_MAXK = 10;
constexpr double RESID_DENSE_FRAC = 0.15;
constexpr double LN2 = 0.6931471805599453;
constexpr double RESID_EXPONENTS[] = {2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0};
constexpr int64_t LONG_Q_ROWS = 24 * 1024;
constexpr int64_t VERY_LONG_MEAN_Q = 4 * 1024;
constexpr int64_t SKEW_Q_ROWS = 2 * 1024;
constexpr int64_t REPLAY_Q_ROWS = 10 * 1024;
constexpr double MCAST_TILE_RATIO = 0.98;
constexpr double TWOCTA_TILES_PER_CTA = 1800.0;
constexpr double LEAD_PIECE = 900.0;
constexpr double DEEP_TILES_PER_CTA = 4000.0;
constexpr double DEEP_PIECE = 500.0;

inline int64_t floordiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

// CPython 3.12 sum() over floats: Neumaier compensated summation.
struct Neu {
  double s = 0.0, c = 0.0;
  inline void add(double x) {
    double t = s + x;
    if (std::fabs(s) >= std::fabs(x)) c += (s - t) + x;
    else c += (x - t) + s;
    s = t;
  }
  inline double get() const { return (c != 0.0 && std::isfinite(c)) ? s + c : s; }
};

inline int mask_k(int64_t npg, double scale, double alpha) {
  double want = scale / std::pow((double)npg, alpha);
  if (want >= 1.0) return 0;
  int k = 0;
  while (k < RESID_MAXK && (1.0 / (double)(1 << (k + 1))) >= want) ++k;
  return k;
}

inline double stream_neff(const double* v, size_t n) {
  double tot = 0.0, sq = 0.0;
  for (size_t i = 0; i < n; ++i) { tot += v[i]; sq += v[i] * v[i]; }
  return sq > 0.0 ? (tot * tot / sq) : 1.0;
}

struct Item {
  int64_t tok_base, n_tok, h, b, causal, npg;
  int32_t sid;  // index of the (b, h) KV stream in first-appearance order
  int32_t mk;   // _mask_k(max(1, npg), rscale) (0 when rscale is None)
  double wm;    // wfun(npg), 1.0 when wfun is None
};

struct Part {
  double cost;
  int32_t item, p0, pw, slot, sb, ns, grp;
};

struct SK {
  double cost;
  int32_t pos, part;
};

struct PackRes {
  double ms = 0.0;
  int64_t nslot = 0, ngrp = 1, nparts = 0, active = 0;
  double nstream = 1.0;
};

struct Cfg {
  int64_t chunk = 0;
  bool use_splits = false;
  std::vector<int64_t> splits;
  bool streamwave = false;
  void set(int64_t c, const std::vector<int64_t>* sp) {
    chunk = c;
    use_splits = sp != nullptr;
    if (sp) splits.assign(sp->begin(), sp->end());
    streamwave = false;
  }
  const int64_t* sp() const { return use_splits ? splits.data() : nullptr; }
};

struct BuildRes {
  PackRes r;
  Cfg cfg;
};

struct Grids {
  const double* kgrid; int64_t nk;
  const double* base; int64_t nb;
  const double* dense; int64_t nd;
  bool filler;
};

constexpr double INF = HUGE_VAL;

// Tournament tree over n slots ordered by (key, index).  Indices are unique, so
// its minimum is exactly what heapq pops from a heap of (key, index) tuples, and
// the pop-then-push-same-index pattern of the Python planner is one leaf update.
struct MinTree {
  struct N { double k; int32_t i; };
  int32_t size = 1;
  std::vector<N> nd;  // nd[1] root; leaves at nd[size + i]
  static inline bool lt(const N& a, const N& b) { return a.k < b.k || (a.k == b.k && a.i < b.i); }
  void init(int64_t n, double fill) {
    size = 1;
    while (size < n) size <<= 1;
    nd.resize(2 * (size_t)size);
    for (int32_t i = 0; i < size; ++i) nd[size + i] = {i < n ? fill : INF, i};
    rebuild();
  }
  void rebuild() {
    for (int32_t v = size - 1; v >= 1; --v) nd[v] = lt(nd[2 * v], nd[2 * v + 1]) ? nd[2 * v] : nd[2 * v + 1];
  }
  inline double& key(int32_t i) { return nd[size + i].k; }  // direct leaf write: rebuild() after
  inline int32_t top() const { return nd[1].i; }
  inline double topkey() const { return nd[1].k; }
  inline void update(int32_t i, double k) {
    int32_t v = size + i;
    double ck = k;
    int32_t ci = i;
    nd[v] = {ck, ci};
    while (v > 1) {
      const N s = nd[v ^ 1];
      // branchless (s < c): the winner is data-dependent and mispredicts badly
      const bool b = (s.k < ck) | ((s.k == ck) & (s.i < ci));
      ck = b ? s.k : ck;
      ci = b ? s.i : ci;
      v >>= 1;
      nd[v] = {ck, ci};
    }
  }
};

struct Out {
  int64_t nwork, nbins, nmrg, nmbin, nslot, ngrp, variant, precise, grid;
  bool use_mcast;
};

struct Planner {
  std::vector<Item> items_flat, items_pair;
  // _pack scratch
  std::vector<Part> parts;
  std::vector<Neu> sacc;
  std::vector<double> sval, gw, freev, ready, ctime;
  std::vector<int32_t> scount, sstart, scur, slist, sorder, binof, bcount, bstart, bcur, bjobs;
  std::vector<int32_t> arrived, pos, gorder, mcount;
  std::vector<std::pair<int32_t, int32_t>> groups;
  std::vector<SK> skey, runs;
  std::vector<int32_t> seqp;
  struct Job { double cost; int32_t ns, grp; };
  std::vector<Job> jobs;
  MinTree tree;
  // Memo of _pack results within one _build_plan call.  The parts (hence the
  // result) are a function of each item's piece length alone, so different
  // chunk / wave / filler configurations that cut every item the same way
  // (e.g. the base waves repeated inside the dense grid) are packed once.
  struct CacheEnt { uint64_t h; size_t off; bool ok; PackRes r; };
  std::vector<CacheEnt> cache;
  std::vector<int32_t> ckeys, pkey;
  std::vector<uint8_t> has;
  std::vector<std::array<int64_t, 5>> info;
  std::vector<std::pair<int32_t, std::array<int32_t, 8>>> mtmp;
  bool merge_err = false;
  // _build_plan scratch
  std::vector<int64_t> sp, fill_order, seen;
  std::vector<double> raw, frac;
  std::vector<int32_t> word;
  Cfg best_cfg, dense_cfg;
  BuildRes flat, pairb;
  // materialized plan
  std::vector<int32_t> rows, rows_item, starts, mrows, mstart;
  // _resid_scale scratch
  struct RE { int64_t npg; int32_t seq; int64_t n, N; };
  std::vector<RE> rent;
  std::vector<int64_t> gnpg, gN;
  std::vector<double> gr0, gr1;

  bool resid_scale(const int32_t* qls, const int32_t* Ls, int64_t B, double f_cur,
                   double& scale_out, double& alpha_out) {
    rent.clear();
    int32_t s = 0;
    for (int64_t b = 0; b < B; ++b) {
      int64_t ql = qls[b], L = Ls[b];
      for (int64_t qt = 0; qt < ql; qt += 16) {
        int64_t n = std::min<int64_t>(16, ql - qt);
        int64_t causal = L - ql + qt;
        int64_t npg = std::max<int64_t>(1, floordiv(causal + n + PAGE - 1, PAGE));
        int64_t N = std::max<int64_t>(causal + n, 1);
        rent.push_back({npg, s++, n, N});
      }
    }
    if (rent.empty()) return false;
    std::sort(rent.begin(), rent.end(), [](const RE& a, const RE& b) {
      return a.npg != b.npg ? a.npg < b.npg : a.seq < b.seq;
    });
    gnpg.clear(); gN.clear(); gr0.clear(); gr1.clear();
    for (const RE& e : rent) {
      double v0 = (double)e.n / (double)e.N;
      double v1 = (double)(e.n * e.npg);
      if (gnpg.empty() || gnpg.back() != e.npg) {
        gnpg.push_back(e.npg); gr0.push_back(v0); gr1.push_back(v1); gN.push_back(e.N);
      } else {
        gr0.back() += v0;
        gr1.back() += v1;
        if (e.N < gN.back()) gN.back() = e.N;
      }
    }
    const size_t G = gnpg.size();
    double ref = (double)gnpg.back();
    Neu a0, a1;
    double worst0 = 0.0;
    for (size_t g = 0; g < G; ++g) {
      a0.add(gr0[g]);
      a1.add(gr1[g]);
      double w = std::sqrt((1.0 - f_cur) / (double)gN[g]);
      if (g == 0 || w > worst0) worst0 = w;
    }
    double varc = a0.get();
    double tiles = a1.get();
    double dense_cap = RESID_DENSE_FRAC * tiles;
    double lim = std::sqrt(1.0 - f_cur);
    double best_cost = tiles * f_cur;
    bool found = false;
    for (double alpha : RESID_EXPONENTS) {
      double base = alpha * std::log(ref);
      for (int e = -40; e <= 40; ++e) {
        double scale = std::exp(base + ((double)e / 2.0) * LN2) / 16.0;
        double rem = 0.0, cost = 0.0, worst = 0.0, dense = 0.0;
        bool bad = false;
        for (size_t g = 0; g < G; ++g) {
          double f = 1.0 / (double)(1 << mask_k(gnpg[g], scale, alpha));
          rem += gr0[g] * f;
          cost += gr1[g] * f;
          if (f > f_cur) dense += gr1[g];
          double om = 1.0 - f;
          double w = std::sqrt((om > 0.0 ? om : 0.0) / (double)gN[g]);
          if (w > worst) worst = w;
          // cost, worst and dense never decrease: once rejected, stay rejected
          if (cost >= best_cost || worst > worst0 || dense > dense_cap) { bad = true; break; }
        }
        if (bad) continue;
        double x = 1.0 - rem / varc;
        if (std::sqrt(x > 0.0 ? x : 0.0) <= lim) {
          scale_out = scale;
          alpha_out = alpha;
          best_cost = cost;
          found = true;
        }
      }
    }
    return found;
  }

  // Parts count of _pack(items, chunk) or false when it returns None.
  bool count_parts(const std::vector<Item>& items, int64_t chunk, int dup, int64_t& np) {
    int64_t nslot = 0;
    np = 0;
    for (const Item& it : items) {
      int64_t ns = floordiv(it.npg + chunk - 1, chunk);
      if (ns <= 1) { ++np; continue; }
      int64_t per = floordiv(it.npg + ns - 1, ns);
      ns = floordiv(it.npg + per - 1, per);
      nslot += dup * ns;
      if (nslot > NSLOT_CAP) return false;
      np += ns;
    }
    return true;
  }

  // _score_bins.  Returns makespan; fills mrows/mstart when mat.
  double score_bins(const std::vector<Item>& items, int64_t nsm, int dup, int64_t ngrp_raw, bool mat) {
    const int64_t ngrid = nsm * dup;
    freev.assign(ngrid, 0.0);
    const int64_t G = ngrp_raw * dup;
    has.assign(G, 0);
    ready.assign(G, 0.0);
    info.resize(G);
    int64_t ninfo = 0;
    for (int64_t i = 0; i < nsm; ++i) {
      double cum = 0.0;
      for (int32_t j = bstart[i]; j < bstart[i + 1]; ++j) {
        const Part& p = parts[bjobs[j]];
        cum += p.cost;
        if (p.ns > 1) {
          const Item& it = items[p.item];
          for (int r = 0; r < dup; ++r) {
            int64_t g = (int64_t)p.grp * dup + r;
            int64_t nt = it.n_tok, tb = it.tok_base;
            if (dup > 1) {
              if (nt > 16) tb = tb + r * 16;
              nt = nt - r * 16;
              if (nt > 16) nt = 16;
            }
            if (!has[g]) {
              has[g] = 1;
              info[g] = {tb, nt, it.h, (int64_t)p.sb + r * p.ns, p.ns};
              ++ninfo;
            }
            if (cum > ready[g]) ready[g] = cum;
          }
        }
      }
      for (int r = 0; r < dup; ++r) freev[i * dup + r] = cum;
    }
    if (mat) { mrows.clear(); mstart.assign(ngrid + 1, 0); }
    if (ninfo == 0) {
      double m = freev[0];
      for (int64_t c = 1; c < ngrid; ++c) if (freev[c] > m) m = freev[c];
      return m;
    }
    if (ngrid < NMRG) { merge_err = true; return 0.0; }
    tree.init(ngrid, 0.0);
    for (int64_t c = 0; c < ngrid; ++c) tree.key((int32_t)c) = freev[c];
    tree.rebuild();
    gorder.clear();
    for (int64_t g = 0; g < G; ++g) if (has[g]) gorder.push_back((int32_t)g);
    std::sort(gorder.begin(), gorder.end(), [this](int32_t a, int32_t b) {
      return ready[a] != ready[b] ? ready[a] < ready[b] : a < b;
    });
    if (mat) mtmp.clear();
    const int64_t norder = (int64_t)gorder.size();
    for (int64_t gi = 0; gi < norder; ++gi) {
      const int64_t rem = norder - gi;
      const int32_t g = gorder[gi];
      const auto& inf = info[g];
      const int64_t ns = inf[4];
      const double band = C_MERGE * (double)ns / (double)NMRG + C_MPRE * (double)ns;
      const double rdy = ready[g];
      int32_t cand[NMRG];
      double cfr[NMRG];
      for (int k = 0; k < NMRG; ++k) {
        cand[k] = tree.top();
        cfr[k] = tree.topkey();
        tree.update(cand[k], INF);
      }
      double bfin = 0.0, bcost = 0.0;
      int64_t ntask = 0;
      for (int64_t nt_try : {1, 2, 4}) {
        double cost_try = band * (double)(NMRG / nt_try) + C_MTASK;
        double fr = cfr[nt_try - 1];
        double fin_try = (fr > rdy ? fr : rdy) + cost_try;
        fin_try += (double)(rem * nt_try) * cost_try / (double)ngrid;
        if (ntask == 0 || fin_try < bfin) { bfin = fin_try; ntask = nt_try; bcost = cost_try; }
      }
      const int64_t nband = NMRG / ntask;
      for (int64_t k = 0; k < ntask; ++k) {
        double fr = cfr[k];
        int32_t c = cand[k];
        double fin = (fr > rdy ? fr : rdy) + bcost;
        if (mat)
          mtmp.push_back({c, {(int32_t)inf[0], (int32_t)inf[1], (int32_t)inf[2], (int32_t)inf[3],
                              (int32_t)ns, g, (int32_t)(k * nband), (int32_t)ntask}});
        freev[c] = fin;
        tree.update(c, fin);
      }
      for (int64_t k = ntask; k < NMRG; ++k) tree.update(cand[k], cfr[k]);
    }
    double m = freev[0];
    for (int64_t c = 1; c < ngrid; ++c) if (freev[c] > m) m = freev[c];
    if (mat) {
      mcount.assign(ngrid, 0);
      for (const auto& e : mtmp) ++mcount[e.first];
      for (int64_t c = 0; c < ngrid; ++c) mstart[c + 1] = mstart[c] + mcount[c];
      mrows.resize(mtmp.size() * MRG_F);
      for (int64_t c = 0; c < ngrid; ++c) mcount[c] = mstart[c];
      for (const auto& e : mtmp) {
        int32_t* o = &mrows[(size_t)(mcount[e.first]++) * MRG_F];
        for (int f = 0; f < MRG_F; ++f) o[f] = e.second[f];
      }
    }
    return m;
  }

  // _score_last.  Only split parts (ns > 1) interact across CTAs, and heapq pops
  // events in increasing (time, cta) order with every time computed by the same
  // per-CTA running sum, so the queue need only hold each CTA's next split part;
  // the unsplit parts in between are summed in order.  Times strictly increase
  // along a CTA, so the makespan is the largest final CTA time.
  double score_last(int64_t nsm, int64_t ngrp_raw) {
    arrived.assign(ngrp_raw, 0);
    pos.assign(nsm, 0);
    ctime.assign(nsm, 0.0);
    tree.init(nsm, INF);
    auto advance = [this](int32_t c) -> double {
      const int32_t end = bcount[c];
      const Job* jb = &jobs[bstart[c]];
      int32_t q = pos[c];
      double t = ctime[c];
      while (q < end && jb[q].ns <= 1) t += jb[q++].cost;
      pos[c] = q;
      ctime[c] = t;
      return q < end ? t + jb[q].cost : INF;
    };
    for (int32_t c = 0; c < nsm; ++c) tree.key(c) = advance(c);
    tree.rebuild();
    for (;;) {
      const int32_t c = tree.top();
      double t = tree.topkey();
      if (t == INF) break;
      const Job& p = jobs[bstart[c] + pos[c]];
      if (++arrived[p.grp] == p.ns) t += C_MERGE * (double)p.ns;
      ++pos[c];
      ctime[c] = t;
      tree.update(c, advance(c));
    }
    double ms = 0.0;
    for (int64_t c = 0; c < nsm; ++c) if (ctime[c] > ms) ms = ctime[c];
    return ms;
  }

  // _pack through the per-build memo (plain, non-materializing packs only).
  bool pack(const std::vector<Item>& items, int64_t chunk, const int64_t* splits, int64_t nsm, int dup,
            bool streamwave, int scoremode, bool mat, PackRes& out) {
    if (streamwave || mat) return pack_impl(items, chunk, splits, nsm, dup, streamwave, scoremode, mat, out);
    const size_t n = items.size();
    pkey.resize(n);
    uint64_t h = 1469598103934665603ULL;
    for (size_t idx = 0; idx < n; ++idx) {
      const int64_t npg = items[idx].npg;
      const int64_t ns = splits ? splits[idx] : floordiv(npg + chunk - 1, chunk);
      const int32_t k = ns <= 1 ? 0 : (int32_t)floordiv(npg + ns - 1, ns);
      pkey[idx] = k;
      h = (h ^ (uint32_t)k) * 1099511628211ULL;
    }
    for (const CacheEnt& e : cache) {
      if (e.h == h && std::equal(pkey.begin(), pkey.end(), ckeys.begin() + e.off)) {
        out = e.r;
        return e.ok;
      }
    }
    const bool ok = pack_impl(items, chunk, splits, nsm, dup, false, scoremode, false, out);
    if (merge_err) return false;
    cache.push_back({h, ckeys.size(), ok, out});
    ckeys.insert(ckeys.end(), pkey.begin(), pkey.end());
    return ok;
  }

  // _pack.  Returns false where _pack returns None.
  bool pack_impl(const std::vector<Item>& items, int64_t chunk, const int64_t* splits, int64_t nsm, int dup,
                 bool streamwave, int scoremode, bool mat, PackRes& out) {
    parts.clear();
    int64_t nslot = 0, grp = 0;
    const int64_t n = (int64_t)items.size();
    for (int64_t idx = 0; idx < n; ++idx) {
      const Item& it = items[idx];
      int64_t ns = splits ? splits[idx] : floordiv(it.npg + chunk - 1, chunk);
      if (ns <= 1) {
        parts.push_back({(double)it.npg * it.wm + C_ITEM, (int32_t)idx, 0, (int32_t)it.npg, -1, 0, 1, -1});
      } else {
        int64_t per = floordiv(it.npg + ns - 1, ns);
        ns = floordiv(it.npg + per - 1, per);
        int64_t sb = nslot;
        nslot += dup * ns;
        if (nslot > NSLOT_CAP) return false;
        for (int64_t s = 0; s < ns; ++s) {
          int64_t p0 = s * per;
          int64_t pw = std::min(per, it.npg - p0);
          parts.push_back({(double)pw * it.wm + C_ITEM + C_SPLIT, (int32_t)idx, (int32_t)p0, (int32_t)pw,
                           (int32_t)(sb + s), (int32_t)sb, (int32_t)ns, (int32_t)grp});
        }
        ++grp;
      }
    }
    const int32_t P = (int32_t)parts.size();
    const int32_t S = items.back().sid + 1;
    sacc.assign(S, Neu());
    scount.assign(S, 0);
    for (const Part& p : parts) {
      int32_t s = items[p.item].sid;
      sacc[s].add(p.cost);
      ++scount[s];
    }
    sval.resize(S);
    for (int32_t s = 0; s < S; ++s) sval[s] = sacc[s].get();
    auto by_cost = [](const SK& a, const SK& b) {
      return a.cost != b.cost ? a.cost > b.cost : a.pos < b.pos;
    };
    double nstream;
    if (!streamwave) {
      // Stable sort by cost, descending.  An item's pieces are consecutive parts
      // and take at most two cost values (full pieces, then a shorter last one),
      // so sorting those runs by (cost desc, first part) is the same order.
      runs.clear();
      for (int32_t j = 0; j < P;) {
        int32_t e = j + 1;
        while (e < P && parts[e].item == parts[j].item && parts[e].cost == parts[j].cost) ++e;
        runs.push_back({parts[j].cost, j, e - j});
        j = e;
      }
      std::sort(runs.begin(), runs.end(), by_cost);
      seqp.resize(P);
      int32_t off = 0;
      for (const SK& r : runs)
        for (int32_t q = 0; q < r.part; ++q) seqp[off++] = r.pos + q;
      nstream = stream_neff(sval.data(), S);
    } else {
      skey.resize(P);
      seqp.resize(P);
      sstart.assign(S + 1, 0);
      for (int32_t s = 0; s < S; ++s) sstart[s + 1] = sstart[s] + scount[s];
      scur.assign(sstart.begin(), sstart.end() - 1);
      slist.resize(P);
      for (int32_t j = 0; j < P; ++j) slist[scur[items[parts[j].item].sid]++] = j;
      sorder.resize(S);
      for (int32_t s = 0; s < S; ++s) sorder[s] = s;
      std::sort(sorder.begin(), sorder.end(), [this](int32_t a, int32_t b) {
        return sval[a] != sval[b] ? sval[a] > sval[b] : a < b;
      });
      groups.clear();
      int64_t cur = 0;
      int32_t gbeg = 0;
      for (int32_t t = 0; t < S; ++t) {
        cur += scount[sorder[t]];
        if (cur >= nsm) { groups.push_back({gbeg, t + 1}); gbeg = t + 1; cur = 0; }
      }
      if (gbeg < S) {
        if (!groups.empty()) groups.back().second = S;
        else groups.push_back({gbeg, S});
      }
      double tot = 0.0, neff = 0.0;
      int32_t off = 0;
      for (const auto& gr : groups) {
        int32_t beg = off;
        gw.clear();
        for (int32_t t = gr.first; t < gr.second; ++t) {
          int32_t s = sorder[t];
          for (int32_t q = sstart[s]; q < sstart[s + 1]; ++q) {
            int32_t j = slist[q];
            skey[off] = {parts[j].cost, off, j};
            ++off;
          }
          gw.push_back(sval[s]);
        }
        std::sort(skey.begin() + beg, skey.begin() + off, by_cost);
        Neu wacc;
        for (double v : gw) wacc.add(v);
        double w = wacc.get();
        tot += w;
        neff += w * stream_neff(gw.data(), gw.size());
      }
      nstream = tot > 0.0 ? (neff / tot) : 1.0;
      for (int32_t j = 0; j < P; ++j) seqp[j] = skey[j].part;
    }
    // LPT over nsm persistent CTAs, waves in order.
    tree.init(nsm, 0.0);
    binof.resize(P);
    bcount.assign(nsm, 0);
    for (int32_t j = 0; j < P; ++j) {
      const int32_t c = tree.top();
      binof[j] = c;
      ++bcount[c];
      tree.update(c, tree.topkey() + parts[seqp[j]].cost);
    }
    bstart.assign(nsm + 1, 0);
    int64_t active = 0;
    for (int64_t i = 0; i < nsm; ++i) {
      bstart[i + 1] = bstart[i] + bcount[i];
      if (bcount[i]) ++active;
    }
    bcur.assign(bstart.begin(), bstart.end() - 1);
    bjobs.resize(P);
    jobs.resize(P);
    for (int32_t j = 0; j < P; ++j) {
      const int32_t q = bcur[binof[j]]++;
      const Part& p = parts[seqp[j]];
      bjobs[q] = seqp[j];
      jobs[q] = {p.cost, p.ns, p.grp};
    }
    double ms;
    if (scoremode || mat) {
      double msb = score_bins(items, nsm, dup, grp, mat);
      if (merge_err) return false;
      ms = msb;
    }
    if (!scoremode) ms = score_last(nsm, grp);
    if (mat) {
      rows.resize((size_t)P * PLAN_F);
      rows_item.resize(P);
      starts.assign(bstart.begin(), bstart.end());
      for (int32_t q = 0; q < P; ++q) {
        const Part& p = parts[bjobs[q]];
        const Item& it = items[p.item];
        int32_t* o = &rows[(size_t)q * PLAN_F];
        o[0] = (int32_t)it.tok_base; o[1] = (int32_t)it.n_tok; o[2] = (int32_t)it.h; o[3] = (int32_t)it.b;
        o[4] = (int32_t)((int64_t)p.p0 * PAGE); o[5] = p.pw; o[6] = (int32_t)it.causal; o[7] = p.slot;
        o[8] = p.sb; o[9] = p.ns; o[10] = p.grp; o[11] = 7;
        rows_item[q] = p.item;
      }
    }
    out.ms = ms;
    out.nslot = nslot;
    out.ngrp = std::max<int64_t>(grp, 1);
    out.nparts = P;
    out.active = active;
    out.nstream = nstream;
    return true;
  }

  bool wave_splits(const std::vector<Item>& items, int64_t nsm, double wave) {
    int64_t total = 0;
    for (const Item& it : items) total += it.npg;
    if (total <= 0) return false;
    const double target = wave * (double)nsm;
    const size_t n = items.size();
    raw.resize(n);
    sp.resize(n);
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
      raw[i] = (double)items[i].npg * target / (double)total;
      sp[i] = std::max<int64_t>(1, std::min<int64_t>(items[i].npg, (int64_t)raw[i]));
      sum += sp[i];
    }
    double shortv = target - (double)sum;
    if (shortv > 0) {
      frac.resize(n);
      word.resize(n);
      for (size_t i = 0; i < n; ++i) { frac[i] = raw[i] - (double)(int64_t)raw[i]; word[i] = (int32_t)i; }
      std::sort(word.begin(), word.end(), [this](int32_t a, int32_t b) {
        return frac[a] != frac[b] ? frac[a] > frac[b] : a < b;
      });
      size_t k = 0;
      while (shortv > 0 && k < n) {
        int32_t i = word[k];
        if (sp[i] < items[i].npg) { sp[i] += 1; shortv -= 1; }
        ++k;
      }
    }
    return true;
  }

  bool filler_splits(const std::vector<Item>& items, int64_t k, int64_t ns) {
    if (k <= 0 || ns <= 1 || items.empty()) return false;
    const size_t n = items.size();
    sp.assign(n, 1);
    bool cut = false;
    const size_t lim = std::min<size_t>((size_t)k, n);
    for (size_t q = 0; q < lim; ++q) {
      int64_t i = fill_order[q];
      if (items[i].npg >= ns) { sp[i] = ns; cut = true; }
    }
    return cut;
  }

  // _build_plan (items non-empty).
  bool build(const std::vector<Item>& items, int64_t nsm_total, bool pair, int scoremode, const Grids& gr,
             BuildRes& out) {
    cache.clear();
    ckeys.clear();
    const int dup = pair ? 2 : 1;
    const int64_t nsm = pair ? std::max<int64_t>(1, nsm_total / 2) : nsm_total;
    Neu tacc;
    for (const Item& it : items) tacc.add((double)it.npg + C_ITEM);
    const double total = tacc.get();
    bool have = false;
    PackRes best, res;
    for (int64_t w = 0; w < gr.nb; ++w) {
      if (!wave_splits(items, nsm, gr.base[w])) continue;
      if (pack(items, 0, sp.data(), nsm, dup, false, scoremode, false, res) && (!have || res.ms < best.ms)) {
        best = res; have = true; best_cfg.set(0, &sp);
      }
      if (merge_err) return false;
    }
    seen.clear();
    int floor_state = -1;
    for (int64_t q = 0; q < gr.nk; ++q) {
      int64_t raw_chunk = std::max<int64_t>(1, (int64_t)std::ceil((total / (double)nsm) / gr.kgrid[q]));
      int64_t chunk = std::max<int64_t>(80, raw_chunk);
      if (chunk != raw_chunk) {
        if (floor_state < 0) {
          int64_t np;
          floor_state = (count_parts(items, chunk, dup, np) && np < nsm) ? 1 : 0;
        }
        if (floor_state == 1) chunk = raw_chunk;
      }
      if (std::find(seen.begin(), seen.end(), chunk) != seen.end()) continue;
      seen.push_back(chunk);
      if (pack(items, chunk, nullptr, nsm, dup, false, scoremode, false, res) && (!have || res.ms < best.ms)) {
        best = res; have = true; best_cfg.set(chunk, nullptr);
      }
      if (merge_err) return false;
    }
    if (!have) {
      int64_t mx = items[0].npg;
      for (const Item& it : items) if (it.npg > mx) mx = it.npg;
      best_cfg.set(mx, nullptr);
      if (!pack(items, mx, nullptr, nsm, dup, false, scoremode, false, best)) return false;
    }
    const PackRes base = best;
    PackRes dense = best;
    dense_cfg = best_cfg;
    for (int64_t w = 0; w < gr.nd; ++w) {
      if (!wave_splits(items, nsm, gr.dense[w])) continue;
      if (pack(items, 0, sp.data(), nsm, dup, false, scoremode, false, res) && res.ms < dense.ms) {
        dense = res; dense_cfg.set(0, &sp);
      }
      if (merge_err) return false;
    }
    if (dense.ms <= base.ms * 0.99 && dense.nslot <= base.nslot &&
        (dense.active >= base.active || dense.ms <= base.ms * 0.95)) {
      best = dense;
      best_cfg = dense_cfg;
    }
    if (gr.filler && (int64_t)items.size() >= nsm) {
      const size_t n = items.size();
      fill_order.resize(n);
      for (size_t i = 0; i < n; ++i) fill_order[i] = (int64_t)i;
      std::sort(fill_order.begin(), fill_order.end(), [&items](int64_t a, int64_t b) {
        return items[a].npg != items[b].npg ? items[a].npg > items[b].npg : a < b;
      });
      const int64_t fks[4] = {std::max<int64_t>(1, nsm / 4), std::max<int64_t>(1, nsm / 2), nsm, 2 * nsm};
      for (int64_t fk : fks) {
        for (int64_t fns : {2, 3, 4, 6, 8}) {
          if (!filler_splits(items, fk, fns)) continue;
          if (pack(items, 0, sp.data(), nsm, dup, false, scoremode, false, res) && res.ms < best.ms * 0.995) {
            best = res; best_cfg.set(0, &sp);
          }
          if (merge_err) return false;
        }
      }
    }
    PackRes alt;
    if (pack(items, best_cfg.chunk, best_cfg.sp(), nsm, dup, true, scoremode, false, alt) &&
        alt.ms <= best.ms * 1.01 && alt.nslot <= best.nslot && best.nstream >= 4.0 &&
        alt.nstream * 2.0 <= best.nstream) {
      best = alt;
      best_cfg.streamwave = true;
    }
    if (merge_err) return false;
    out.r = best;
    out.cfg = best_cfg;
    return true;
  }

  void make_items(const int32_t* qls, const int32_t* Ls, int64_t B, int64_t step, bool rs, double rsc,
                  double ral, double wnorm, std::vector<Item>& items) {
    items.clear();
    int64_t off = 0;
    int32_t nsid = 0;
    for (int64_t b = 0; b < B; ++b) {
      int64_t ql = qls[b], L = Ls[b];
      if (ql > 0) {
        for (int64_t qt = 0; qt < ql; qt += step) {
          int64_t n_tok = std::min<int64_t>(step, ql - qt);
          int64_t causal = L - ql + qt;
          int64_t npg = floordiv(causal + n_tok + PAGE - 1, PAGE);
          int32_t mk = rs ? mask_k(std::max<int64_t>(1, npg), rsc, ral) : 0;
          double wm = rs ? (1.0 + RESID_ALPHA / (double)(1 << mk)) / wnorm : 1.0;
          for (int64_t h = 0; h < HKV; ++h)
            items.push_back({off + qt, n_tok, h, b, causal, npg, (int32_t)(nsid + h), mk, wm});
        }
        nsid += HKV;
      }
      off += ql;
    }
  }

  Out run(const int32_t* qls, const int32_t* Ls, int64_t B, int64_t nsm, const Grids& gr, int32_t* plan_out,
          int64_t cap_work, int32_t* bins_out, int64_t cap_bins, int32_t* mrg_out, int64_t cap_mrg,
          int32_t* mbin_out, int64_t cap_mbin) {
    Out o{};
    merge_err = false;
    int64_t total_q = 0, max_q = 0, maxL = 0;
    bool bad = nsm < 1;
    for (int64_t b = 0; b < B; ++b) {
      if (qls[b] < 0 || Ls[b] < qls[b]) bad = true;
      total_q += qls[b];
      if (b == 0 || qls[b] > max_q) max_q = qls[b];
      if (b == 0 || Ls[b] > maxL) maxL = Ls[b];
    }
    if (bad) { o.nwork = -3; return o; }
    const double f_cur = total_q >= LONG_Q_ROWS ? (1.0 / 16.0) : (1.0 / 8.0);
    double rsc = 0.0, ral = 0.0;
    const bool rs = resid_scale(qls, Ls, B, f_cur, rsc, ral);
    int kflat = 0;
    while ((1.0 / (double)(1 << (kflat + 1))) >= f_cur) ++kflat;
    const double wnorm = 1.0 + RESID_ALPHA * f_cur;
    make_items(qls, Ls, B, 16, rs, rsc, ral, wnorm, items_flat);
    if (items_flat.empty()) { o.nwork = -2; return o; }
    make_items(qls, Ls, B, 32, rs, rsc, ral, wnorm, items_pair);
    const bool has_pair = nsm >= 2;
    if (!build(items_flat, nsm, false, 0, gr, flat) ||
        (has_pair && !build(items_pair, nsm, true, 0, gr, pairb))) { o.nwork = -3; return o; }
    const bool pick_pair = has_pair && pairb.r.ms * MCAST_TILE_RATIO < flat.r.ms;
    const int64_t nbin0 = pick_pair ? nsm / 2 : nsm;
    const int64_t act0 = pick_pair ? pairb.r.active : flat.r.active;
    if (act0 * 4 < nbin0 * 3) {
      if (!build(items_flat, nsm, false, 1, gr, flat) ||
          (has_pair && !build(items_pair, nsm, true, 1, gr, pairb))) { o.nwork = -3; return o; }
    }
    const bool use_mcast = has_pair && pairb.r.ms * MCAST_TILE_RATIO < flat.r.ms;
    const BuildRes& ch = use_mcast ? pairb : flat;
    const std::vector<Item>& items = use_mcast ? items_pair : items_flat;
    const int dup = use_mcast ? 2 : 1;
    const int64_t nbin = use_mcast ? std::max<int64_t>(1, nsm / 2) : nsm;
    PackRes fin;
    if (!pack(items, ch.cfg.chunk, ch.cfg.sp(), nbin, dup, ch.cfg.streamwave, 1, true, fin)) {
      o.nwork = -3; return o;
    }
    const int64_t nwork = fin.nparts;
    const int64_t nslot = std::max<int64_t>(fin.nslot, 1);
    const int64_t ngrp = fin.ngrp;
    int64_t tiles_tot = 0;
    Neu t1, t2;
    for (int64_t q = 0; q < nwork; ++q) {
      int64_t pw = rows[q * PLAN_F + 5];
      tiles_tot += pw;
      double t = (double)pw;
      t2.add(t * t);
      t1.add(t);
    }
    double tsum = t1.get();
    const double piece_len = t2.get() / (tsum >= 1.0 ? tsum : 1.0);
    const bool deep = use_mcast && piece_len >= DEEP_PIECE;
    const double tiles_per_cta = (double)tiles_tot / (double)std::max<int64_t>(nsm / 2, 1);
    const bool cta2_underfilled = total_q < 6000 && tiles_per_cta >= 1500.0;
    const bool cta2_long_prefix_tail = total_q < 3000 && maxL >= 160000 && tiles_per_cta >= 1100.0;
    const bool cta2_single_short = B == 1 && total_q <= 128 && maxL >= 8192;
    const bool cta2_long_query = tiles_per_cta >= 600.0 && total_q >= 2048;
    int64_t mode;
    if (use_mcast) {
      if (tiles_per_cta >= DEEP_TILES_PER_CTA) mode = 4;
      else if (tiles_per_cta >= TWOCTA_TILES_PER_CTA || cta2_underfilled || cta2_long_prefix_tail ||
               cta2_single_short || cta2_long_query)
        mode = piece_len >= LEAD_PIECE ? 5 : 3;
      else mode = deep ? 2 : 1;
    } else {
      mode = 0;
    }
    const int64_t mode_base = 10 * mode;
    int64_t variant = mode_base + (rs ? 0 : 4);
    if (total_q >= LONG_Q_ROWS) variant = mode_base + (total_q >= VERY_LONG_MEAN_Q * B ? 2 : 1);
    const bool balanced_cta2_replay = mode >= 3 && total_q >= 2 * 1024 && (B == 1 || 2 * max_q <= total_q);
    if (total_q >= REPLAY_Q_ROWS || balanced_cta2_replay) variant += 5;
    int64_t precise = mode_base + 3;
    if (total_q <= SKEW_Q_ROWS) { variant += NOPS; precise += NOPS; }
    if (total_q >= LONG_Q_ROWS) { variant += 2 * NOPS; precise += 2 * NOPS; }
    const int64_t grid = use_mcast ? (nsm / 2) * 2 : nsm;
    const int64_t nmrg_rows = (int64_t)mrows.size() / MRG_F;
    o.nwork = nwork;
    o.nbins = nbin + 1;
    o.nmrg = std::max<int64_t>(nmrg_rows, 1);
    o.nmbin = (int64_t)mstart.size();
    o.nslot = nslot;
    o.ngrp = ngrp;
    o.variant = variant;
    o.precise = precise;
    o.grid = grid;
    o.use_mcast = use_mcast;
    if (nwork > cap_work || o.nbins > cap_bins || o.nmrg > cap_mrg || o.nmbin > cap_mbin) {
      o.nwork = -1;
      return o;
    }
    const int32_t flat11 = (1 << kflat) - 1;
    for (int64_t q = 0; q < nwork; ++q) {
      int32_t* d = plan_out + q * PLAN_F;
      const int32_t* s = &rows[q * PLAN_F];
      for (int f = 0; f < 11; ++f) d[f] = s[f];
      d[11] = rs ? ((1 << items[rows_item[q]].mk) - 1) : flat11;
    }
    for (int64_t i = 0; i < o.nbins; ++i) bins_out[i] = starts[i];
    if (nmrg_rows == 0) {
      for (int f = 0; f < MRG_F; ++f) mrg_out[f] = 0;
    } else {
      std::copy(mrows.begin(), mrows.end(), mrg_out);
    }
    for (int64_t i = 0; i < o.nmbin; ++i) mbin_out[i] = mstart[i];
    return o;
  }
};

void check_i32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_f64(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == torch::kFloat64 && t.is_contiguous() && t.dim() == 1,
              name, " must be a contiguous 1-D float64 CPU tensor");
}

}  // namespace

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool> plan_into(
    torch::Tensor qls, torch::Tensor Ls, int64_t nsm, torch::Tensor plan_out, torch::Tensor bins_out,
    torch::Tensor mrg_out, torch::Tensor mbin_out, torch::Tensor kgrid, torch::Tensor base_waves,
    torch::Tensor dense_waves, bool filler) {
  check_i32(qls, "qls");
  check_i32(Ls, "Ls");
  check_i32(plan_out, "plan_out");
  check_i32(bins_out, "bins_out");
  check_i32(mrg_out, "mrg_out");
  check_i32(mbin_out, "mbin_out");
  check_f64(kgrid, "kgrid");
  check_f64(base_waves, "base_waves");
  check_f64(dense_waves, "dense_waves");
  TORCH_CHECK(qls.dim() == 1 && Ls.dim() == 1 && qls.numel() == Ls.numel(), "qls/Ls must be 1-D, same length");
  TORCH_CHECK(plan_out.dim() == 2 && plan_out.size(1) == PLAN_F, "plan_out must be [cap_work, 12]");
  TORCH_CHECK(mrg_out.dim() == 2 && mrg_out.size(1) == MRG_F, "mrg_out must be [cap_mrg, 8]");
  const Grids gr{kgrid.data_ptr<double>(), kgrid.numel(), base_waves.data_ptr<double>(), base_waves.numel(),
                 dense_waves.data_ptr<double>(), dense_waves.numel(), filler};
  const int32_t* q = qls.data_ptr<int32_t>();
  const int32_t* L = Ls.data_ptr<int32_t>();
  const int64_t B = qls.numel();
  int32_t* po = plan_out.data_ptr<int32_t>();
  int32_t* bo = bins_out.data_ptr<int32_t>();
  int32_t* mo = mrg_out.data_ptr<int32_t>();
  int32_t* mbo = mbin_out.data_ptr<int32_t>();
  const int64_t cw = plan_out.size(0), cb = bins_out.numel(), cm = mrg_out.size(0), cmb = mbin_out.numel();
  Out o;
  {
    pybind11::gil_scoped_release nogil;
    thread_local Planner planner;
    o = planner.run(q, L, B, nsm, gr, po, cw, bo, cb, mo, cm, mbo, cmb);
  }
  return {o.nwork, o.nbins, o.nmrg, o.nmbin, o.nslot, o.ngrp, o.variant, o.precise, o.grid, o.use_mcast};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan_into", &plan_into, "Exact C++ port of the KF prefill planner (setup() host plan)");
}
"""

_MOD: Any = None
_GRIDS: dict[tuple, torch.Tensor] = {}


def build(build_dir=None):
    """JIT-build (once per process) and return the extension module."""
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline

        from vllm import envs

        digest = hashlib.sha256(_CPP.encode()).hexdigest()[:16]
        bd = (
            build_dir
            or os.environ.get("KF_PLANNER_BUILD_DIR")
            or os.path.join(envs.VLLM_CACHE_ROOT, "kf_prefill_attn_planner", digest)
        )
        os.makedirs(bd, exist_ok=True)
        _MOD = load_inline(
            name=f"kf_planner_cpp_{digest}",
            cpp_sources=[_CPP],
            extra_cflags=["-O3", "-ffp-contract=off", "-fno-fast-math"],
            with_cuda=False,
            verbose=False,
            build_directory=bd,
        )
    return _MOD


def _grid(t):
    g = _GRIDS.get(t)
    if g is None:
        g = torch.tensor([float(x) for x in t], dtype=torch.float64)
        _GRIDS[t] = g
    return g


def _i32(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, int):
        x = [x]
    return torch.tensor(x, dtype=torch.int32)


def plan_into(
    qls,
    Ls,
    nsm,
    plan_out,
    bins_out,
    mrg_out,
    mbin_out,
    kgrid=KGRID,
    base_waves=BASE_WAVE_GRID,
    dense_waves=DENSE_WAVE_GRID,
    filler=True,
):
    """Plan one launch into caller-owned int32 CPU tensors; see module docstring."""
    m = _MOD if _MOD is not None else build()
    return m.plan_into(
        _i32(qls),
        _i32(Ls),
        int(nsm),
        plan_out,
        bins_out,
        mrg_out,
        mbin_out,
        _grid(tuple(kgrid)),
        _grid(tuple(base_waves)),
        _grid(tuple(dense_waves)),
        bool(filler),
    )
