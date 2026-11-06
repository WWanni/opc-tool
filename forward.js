/*
 * forward.js -- OPC 도구 forward model (D01~D05) JavaScript 구현.
 * python src/imaging.py 와 동일: I = |F^-1[F[pad(M)] · P]|², P = 원형 pupil (|f| ≤ NA/λ), Z = (I ≥ I_th).
 * pad = round(4 λ/NA / pixel) = 112, FFT 크기는 임의 (Bluestein), 주파수 격자는 fftfreq 와 동일.
 * 검증: webapp/test/verify.js (python 참조값과 비교).
 */
(function (root) {
  "use strict";

  // ---------------- FFT (복소, 길이 임의) ----------------
  const fftCache = new Map();

  function fftPow2(re, im, inverse) {           // in-place radix-2, n = 2^k
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {        // bit reversal
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inverse ? 2 : -2) * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }

  function bluesteinPlan(n) {
    if (fftCache.has(n)) return fftCache.get(n);
    let m = 1; while (m < 2 * n - 1) m <<= 1;
    const wr = new Float64Array(n), wi = new Float64Array(n);   // chirp w_k = exp(-iπk²/n)
    for (let k = 0; k < n; k++) { const a = -Math.PI * ((k * k) % (2 * n)) / n; wr[k] = Math.cos(a); wi[k] = Math.sin(a); }
    const br = new Float64Array(m), bi = new Float64Array(m);   // b_k = conj(w_k), 대칭 배치
    br[0] = wr[0]; bi[0] = -wi[0];
    for (let k = 1; k < n; k++) { br[k] = br[m - k] = wr[k]; bi[k] = bi[m - k] = -wi[k]; }
    fftPow2(br, bi, false);
    const plan = { n, m, wr, wi, br, bi };
    fftCache.set(n, plan); return plan;
  }

  function fft1d(re, im, inverse) {              // in-place, 길이 임의
    const n = re.length;
    if ((n & (n - 1)) === 0) { fftPow2(re, im, inverse); return; }
    if (inverse) {                               // ifft(x) = conj(fft(conj(x))) / n
      for (let k = 0; k < n; k++) im[k] = -im[k];
      fft1d(re, im, false);
      for (let k = 0; k < n; k++) { re[k] /= n; im[k] = -im[k] / n; }
      return;
    }
    const p = bluesteinPlan(n), m = p.m;
    const ar = new Float64Array(m), ai = new Float64Array(m);
    for (let k = 0; k < n; k++) {                // a_k = x_k · w_k
      ar[k] = re[k] * p.wr[k] - im[k] * p.wi[k]; ai[k] = re[k] * p.wi[k] + im[k] * p.wr[k];
    }
    fftPow2(ar, ai, false);
    for (let k = 0; k < m; k++) { const r = ar[k] * p.br[k] - ai[k] * p.bi[k]; ai[k] = ar[k] * p.bi[k] + ai[k] * p.br[k]; ar[k] = r; }
    fftPow2(ar, ai, true);
    for (let k = 0; k < n; k++) {                // X_k = w_k · (a ⊛ b)_k
      re[k] = ar[k] * p.wr[k] - ai[k] * p.wi[k]; im[k] = ar[k] * p.wi[k] + ai[k] * p.wr[k];
    }
  }

  function fft2d(re, im, H, W, inverse) {        // re, im: Float64Array(H*W), in-place
    const rr = new Float64Array(W), ri = new Float64Array(W);
    for (let y = 0; y < H; y++) {
      const o = y * W;
      for (let x = 0; x < W; x++) { rr[x] = re[o + x]; ri[x] = im[o + x]; }
      fft1d(rr, ri, inverse);
      for (let x = 0; x < W; x++) { re[o + x] = rr[x]; im[o + x] = ri[x]; }
    }
    const cr = new Float64Array(H), ci = new Float64Array(H);
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) { cr[y] = re[y * W + x]; ci[y] = im[y * W + x]; }
      fft1d(cr, ci, inverse);
      for (let y = 0; y < H; y++) { re[y * W + x] = cr[y]; im[y * W + x] = ci[y]; }
    }
  }

  // ---------------- 광학 ----------------
  const DEFAULT = { lambda_nm: 13.5, NA: 0.55, pixel_nm: 0.875, I_th: 0.3709 };

  function fftfreq(n, d) {                       // numpy/torch fftfreq 순서
    const f = new Float64Array(n), c = Math.floor((n - 1) / 2) + 1;
    for (let i = 0; i < c; i++) f[i] = i / (n * d);
    for (let i = c; i < n; i++) f[i] = (i - n) / (n * d);
    return f;
  }

  const pupilCache = new Map();
  function pupil(H, W, cfg) {
    const key = `${H}x${W}:${cfg.lambda_nm}:${cfg.NA}:${cfg.pixel_nm}`;
    if (pupilCache.has(key)) return pupilCache.get(key);
    const fy = fftfreq(H, cfg.pixel_nm), fx = fftfreq(W, cfg.pixel_nm), fc2 = (cfg.NA / cfg.lambda_nm) ** 2;
    const P = new Uint8Array(H * W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) P[y * W + x] = (fx[x] * fx[x] + fy[y] * fy[y] <= fc2) ? 1 : 0;
    pupilCache.set(key, P); return P;
  }

  /** mask: Float32Array|Uint8Array (H*W), 0~1. 반환 Float32Array I (H*W). */
  function aerialImage(mask, H, W, cfg) {
    cfg = Object.assign({}, DEFAULT, cfg || {});
    const pad = Math.round(4 * cfg.lambda_nm / cfg.NA / cfg.pixel_nm);
    const Hp = H + 2 * pad, Wp = W + 2 * pad;
    const re = new Float64Array(Hp * Wp), im = new Float64Array(Hp * Wp);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) re[(y + pad) * Wp + x + pad] = mask[y * W + x];
    fft2d(re, im, Hp, Wp, false);
    const P = pupil(Hp, Wp, cfg);
    for (let i = 0; i < Hp * Wp; i++) if (!P[i]) { re[i] = 0; im[i] = 0; }
    fft2d(re, im, Hp, Wp, true);
    const I = new Float32Array(H * W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y + pad) * Wp + x + pad; I[y * W + x] = re[i] * re[i] + im[i] * im[i]; }
    return I;
  }

  function develop(I, I_th) { const Z = new Uint8Array(I.length); for (let i = 0; i < I.length; i++) Z[i] = I[i] >= I_th ? 1 : 0; return Z; }

  /** 목표 대 결과: IoU, 불일치 면적 (missing = 목표에 있고 결과에 없음, extra 반대) [nm²] */
  function compare(Z, Zt, pixel_nm) {
    let inter = 0, uni = 0, miss = 0, extra = 0, tgt = 0;
    for (let i = 0; i < Z.length; i++) {
      const a = Z[i], b = Zt[i];
      if (a && b) inter++; if (a || b) uni++; if (b && !a) miss++; if (a && !b) extra++; if (b) tgt++;
    }
    const a2 = pixel_nm * pixel_nm;
    return { iou: uni ? inter / uni : 1, missing_nm2: miss * a2, extra_nm2: extra * a2, mismatch_nm2: (miss + extra) * a2, target_nm2: tgt * a2 };
  }

  /** 수평 cut-line (행 y) 에서 x 구간 [x0,x1) 의 연속 1 길이 [nm] (첫 run) */
  function cdAlongRow(Z, H, W, y, x0, x1, pixel_nm) {
    let best = 0, run = 0;
    for (let x = x0; x < x1; x++) { if (Z[y * W + x]) { run++; best = Math.max(best, run); } else run = 0; }
    return best * pixel_nm;
  }

  const api = { aerialImage, develop, compare, cdAlongRow, fft1d, fft2d, DEFAULT };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.OPC = api;
})(typeof window !== "undefined" ? window : globalThis);
