// dense_i8.h
#include <adf.h>
#include <cstring>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

// -------------------- compile-time tiling & flags --------------------
static constexpr unsigned TmC = (mm_M + mm_m - 1) / mm_m;
static constexpr unsigned TkC = (mm_K + mm_k - 1) / mm_k;
static constexpr unsigned TnC = (mm_N + mm_n - 1) / mm_n;

static constexpr bool HAS_M_TAIL = (mm_M % mm_m) != 0;
static constexpr bool HAS_K_TAIL = (mm_K % mm_k) != 0;
static constexpr bool HAS_N_TAIL = (mm_N % mm_n) != 0;

static constexpr unsigned VM_LAST = HAS_M_TAIL ? (mm_M - (TmC - 1) * mm_m) : mm_m;
static constexpr unsigned VK_LAST = HAS_K_TAIL ? (mm_K - (TkC - 1) * mm_k) : mm_k;
static constexpr unsigned VN_LAST = HAS_N_TAIL ? (mm_N - (TnC - 1) * mm_n) : mm_n;

static constexpr bool kDoRelu = DO_RELU;

// -------------------- generic helpers (minimal templates) --------------------
// Load a ROWS×COLS tile from src into a vector of size VEC_SIZE, zero-padding if partial.
// FULL_ROWS/FULL_COLS are the full tile dims; ROW_STRIDE is elements per row in the packed tile.
template<unsigned ROWS, unsigned COLS,
         unsigned FULL_ROWS, unsigned FULL_COLS,
         unsigned ROW_STRIDE, unsigned VEC_SIZE>
static inline aie::vector<int8, VEC_SIZE> load_tile(const int8* src) {
  if constexpr (ROWS == FULL_ROWS && COLS == FULL_COLS) {
    return aie::load_v<VEC_SIZE>(src);
  } else {
    alignas(32) int8 buf[VEC_SIZE];
    std::memset(buf, 0, sizeof(buf));
    for (unsigned r = 0; r < ROWS; ++r) {
      std::memcpy(buf + r * ROW_STRIDE, src + r * ROW_STRIDE, COLS);
    }
    return aie::load_v<VEC_SIZE>(buf);
  }
}

// Store a VM×VN tile from vector v to dst, masking rows/cols if partial.
template<unsigned VM, unsigned VN,
         unsigned FULL_M, unsigned FULL_N,
         unsigned ROW_STRIDE, unsigned VEC_SIZE>
static inline void store_tile(int8* dst, aie::vector<int8, VEC_SIZE> v) {
  if constexpr (VM == FULL_M && VN == FULL_N) {
    aie::store_v(dst, v);
  } else {
    alignas(32) int8 buf[VEC_SIZE];
    aie::store_v(buf, v);
    for (unsigned r = 0; r < VM; ++r) {
      std::memcpy(dst + r * ROW_STRIDE, buf + r * ROW_STRIDE, VN);
    }
  }
}

// Post-process (shift -> int8, optional ReLU) and store with masking.
template<unsigned VM, unsigned VN, typename MMUL>
static inline void finalize_store_tile(int8* dst, const MMUL& C) {
  auto v = C.template to_vector<int8>(SHIFT);
  if constexpr (kDoRelu) v = aie::max(v, (int8)0);
  store_tile<VM, VN, mm_m, mm_n, mm_n, MMUL::size_C>(dst, v);
}

// Compute one output tile C(im,in) (size VM×VN) over all K slices (last K may be partial).
template<unsigned VM, unsigned VN, typename MMUL>
static inline MMUL dot_tile(const int8* __restrict pA_base,
                            const int8* __restrict pB_base,
                            unsigned im, unsigned in)
{
  MMUL C;
  bool init = false;

  // Full K-slices (if any): use mm_k
  for (unsigned ik = 0; ik + 1 < TkC; ++ik) chess_flatten_loop {
    const int8* __restrict pA = pA_base + (im * TkC + ik) * MMUL::size_A;
    const int8* __restrict pB = pB_base + (ik * TnC + in) * MMUL::size_B;

    auto A = load_tile<VM, mm_k, mm_m, mm_k, mm_k, MMUL::size_A>(pA);
    auto B = load_tile<mm_k, VN, mm_k, mm_n, mm_n, MMUL::size_B>(pB);

    if (!init) { C.mul(A, B); init = true; } else { C.mac(A, B); }
  }

  // Last K-slice (exists if TkC >= 1): may be partial
  if constexpr (TkC >= 1) {
    const unsigned ik = TkC - 1;
    const int8* __restrict pA = pA_base + (im * TkC + ik) * MMUL::size_A;
    const int8* __restrict pB = pB_base + (ik * TnC + in) * MMUL::size_B;

    if constexpr (!HAS_K_TAIL) {
      auto A = load_tile<VM, mm_k, mm_m, mm_k, mm_k, MMUL::size_A>(pA);
      auto B = load_tile<mm_k, VN, mm_k, mm_n, mm_n, MMUL::size_B>(pB);
      if (!init) C.mul(A, B); else C.mac(A, B);
    } else {
      auto A = load_tile<VM, VK_LAST, mm_m, mm_k, mm_k, MMUL::size_A>(pA);
      auto B = load_tile<VK_LAST, VN, mm_k, mm_n, mm_n, MMUL::size_B>(pB);
      if (!init) C.mul(A, B); else C.mac(A, B);
    }
  }

  return C;
}

// -------------------------------- main kernel --------------------------------
void dense_i8(
  input_window_int8 * __restrict matA,
  output_window_int8 * __restrict matC
){
  using MMUL = aie::mmul<mm_m, mm_k, mm_n, int8, int8>;

  const int8* __restrict pA_base = (const int8*)matA->ptr; // tiles (im,ik)
  const int8* __restrict pB_base = (const int8*)matB;      // tiles (ik,in)
  int8*       __restrict pC_base = (int8*)matC->ptr;       // tiles (im,in)

#ifdef TILE_PROFILING
  aie::tile t = aie::tile::current();
  uint64 c0 = t.cycles();
#endif

  // 2x2 full-tile blocks
  for (unsigned im = 0; im + 1 < TmC; im += 2) chess_flatten_loop
  {
    for (unsigned in = 0; in + 1 < TnC; in += 2) chess_flatten_loop
    {
      auto C00 = dot_tile<mm_m, mm_n, MMUL>(pA_base, pB_base, im + 0, in + 0);
      auto C01 = dot_tile<mm_m, mm_n, MMUL>(pA_base, pB_base, im + 0, in + 1);
      auto C10 = dot_tile<mm_m, mm_n, MMUL>(pA_base, pB_base, im + 1, in + 0);
      auto C11 = dot_tile<mm_m, mm_n, MMUL>(pA_base, pB_base, im + 1, in + 1);

      int8* __restrict pC00 = pC_base + ((im + 0) * TnC + (in + 0)) * MMUL::size_C;
      int8* __restrict pC01 = pC_base + ((im + 0) * TnC + (in + 1)) * MMUL::size_C;
      int8* __restrict pC10 = pC_base + ((im + 1) * TnC + (in + 0)) * MMUL::size_C;
      int8* __restrict pC11 = pC_base + ((im + 1) * TnC + (in + 1)) * MMUL::size_C;

      finalize_store_tile<mm_m, mm_n, MMUL>(pC00, C00);
      finalize_store_tile<mm_m, mm_n, MMUL>(pC01, C01);
      finalize_store_tile<mm_m, mm_n, MMUL>(pC10, C10);
      finalize_store_tile<mm_m, mm_n, MMUL>(pC11, C11);
    }

    // Tail in N: 2x1
    if constexpr (HAS_N_TAIL) {
      const unsigned in = TnC - 1;
      auto C0 = dot_tile<mm_m, VN_LAST, MMUL>(pA_base, pB_base, im + 0, in);
      auto C1 = dot_tile<mm_m, VN_LAST, MMUL>(pA_base, pB_base, im + 1, in);

      int8* __restrict pC0 = pC_base + ((im + 0) * TnC + in) * MMUL::size_C;
      int8* __restrict pC1 = pC_base + ((im + 1) * TnC + in) * MMUL::size_C;

      finalize_store_tile<mm_m, VN_LAST, MMUL>(pC0, C0);
      finalize_store_tile<mm_m, VN_LAST, MMUL>(pC1, C1);
    }
  }

  // Tail in M: 1x2 and (optional) 1x1
  if constexpr (HAS_M_TAIL) {
    const unsigned im = TmC - 1;

    for (unsigned in = 0; in + 1 < TnC; in += 2) chess_flatten_loop
    {
      auto C0 = dot_tile<VM_LAST, mm_n, MMUL>(pA_base, pB_base, im, in + 0);
      auto C1 = dot_tile<VM_LAST, mm_n, MMUL>(pA_base, pB_base, im, in + 1);

      int8* __restrict pC0 = pC_base + (im * TnC + (in + 0)) * MMUL::size_C;
      int8* __restrict pC1 = pC_base + (im * TnC + (in + 1)) * MMUL::size_C;

      finalize_store_tile<VM_LAST, mm_n, MMUL>(pC0, C0);
      finalize_store_tile<VM_LAST, mm_n, MMUL>(pC1, C1);
    }

    if constexpr (HAS_N_TAIL) {
      const unsigned in = TnC - 1;
      auto C = dot_tile<VM_LAST, VN_LAST, MMUL>(pA_base, pB_base, im, in);
      int8* __restrict pC = pC_base + (im * TnC + in) * MMUL::size_C;
      finalize_store_tile<VM_LAST, VN_LAST, MMUL>(pC, C);
    }
  }

#ifdef TILE_PROFILING
  // -------------------- stats --------------------
  uint64 c1 = t.cycles();
  uint64 cycles = c1 - c0;
  uint64 macs = (uint64)mm_M * (uint64)mm_K * (uint64)mm_N;
  uint64 cycles_expected = macs / 128; // 128 int8 MACs/cycle per MMUL
  double efficiency = 100.0 * (double)cycles_expected / (double)cycles;

  printf("\n\n[dense_i8 2x2 + no-pad] eff=%.1f%%, cycles=%llu, exp=%llu "
         "(mm_m=%d mm_n=%d mm_k=%d  Tm=%u Tk=%u Tn=%u SHIFT=%d)\n",
         efficiency, cycles, cycles_expected, mm_m, mm_n, mm_k, TmC, TkC, TnC, SHIFT);
#endif
}
