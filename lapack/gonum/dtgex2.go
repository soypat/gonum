// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// DTGEX2 swaps adjacent diagonal blocks (A11, B11) and (A22, B22)
// of size 1-by-1 or 2-by-2 in an upper (quasi) triangular matrix pair
// (A, B) by an orthogonal equivalence transformation.
// (A, B) must be in generalized real Schur canonical form (as returned
// by DGGES), i.e. A is block upper triangular with 1-by-1 and 2-by-2
// diagonal blocks. B is upper triangular.
// Optionally, the matrices Q and Z of generalized Schur vectors are
// updated.
//
//	Q(in) * A(in) * Z(in)ᵀ = Q(out) * A(out) * Z(out)ᵀ
//	Q(in) * B(in) * Z(in)ᵀ = Q(out) * B(out) * Z(out)ᵀ
//
// Parameters:
//   - Output bool `illConditioned` is true if the transformed matrix pair (A,B)
//     would be too far from generalized Schur form; (A,B) and (Q,Z) left unchanged.
//   - Output bool `lworkTooSmall` is true if the given work slice is too small. Appropiate value
//     for lwork is stored in work[0].
//   - work is of length at least max( 1, n*(n2+n1), 2*(n1+n2)² )
//   - wantq, wantz indicate whether to update Q or Z transformation matrices, respectively.
//   - j1 is index of first block (A11, B11). 1 <= j1 <= n.
//   - n1 is order of first block (A11, B11). n1 is 0, 1 or 2.
//   - n2 is order of second block (A22, B22). n2 is 0, 1 or 2.
//
// Dtgex2 is an internal routine. It is exported for testing purposes only.
func (impl Implementation) Dtgex2(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int, j1, n1, n2 int, work []float64) (illConditioned, lworkTooSmall bool) {
	// TODO(soypat): some Dlaset(All,0,0) calls on arrays could be replaced with
	// array = [ldst*ldst]float64{}
	const (
		ldst         = 4
		eps          = dlamchP
		smlnum       = dlamchS / eps
		threshFactor = 20.0 // See LAPACK's forum post 1783.
	)
	m := n1 + n2
	lwork := len(work)
	switch {
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case ldq < 1 || (wantq && ldq < n):
		panic(badLdQ)
	case ldz < 1 || (wantz && ldz < n):
		panic(badLdZ)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case j1 < 0 || j1 >= n:
		panic(badJ1)
	case n1 < 0 || n1 > 2:
		panic(badN1)
	case n2 < 0 || n2 > 2:
		panic(badN2)
	case lwork < 1:
		panic(badLWork)
	case n == 0 || n1 > n || j1+n1 > n:
		return false, false // Quick return.
	case lwork < max(1, max(n*m, 2*m*m)):
		work[0] = float64(max(1, max(n*m, 2*m*m)))
		lworkTooSmall = true
		return false, lworkTooSmall
	}
	var (
		iwork                                              [ldst]int
		ir, ircop, li, licop, s, scpy, t, taul, taur, tcpy [ldst * ldst]float64
		bQra21, bRqa21, f, g, sa, sb, scale                float64
		weak, strong                                       bool
		linfo                                              int
	)
	bi := blas64.Implementation()
	wands := true
	// Make a local copy of selected block.
	// Go by default zeros all newly allocated memory.
	// impl.Dlaset(blas.All, ldst, ldst, 0, 0, li[:], ldst)
	// impl.Dlaset(blas.All, ldst, ldst, 0, 0, ir[:], ldst)
	impl.Dlacpy(blas.All, m, m, a[j1*lda+j1:], lda, s[:], ldst)
	impl.Dlacpy(blas.All, m, m, b[j1*ldb+j1:], ldb, t[:], ldst)

	// Intrinsic work vector structure.
	var (
		ldw     = m
		woffset = m * m
	)

	// Compute threshold for testing acceptance of swapping.
	dscale := 0.0
	dsum := 1.0

	impl.Dlacpy(blas.All, m, m, s[:], ldst, work, ldw)
	dscale, dsum = impl.Dlassq(m*m, work, ldw, dscale, dsum)

	dnorma := dscale * math.Sqrt(dsum)
	dscale = 0
	dsum = 1
	impl.Dlacpy(blas.All, m, m, t[:], ldst, work, ldw)
	dscale, dsum = impl.Dlassq(m*m, work, ldw, dscale, dsum)
	dnormb := dscale * math.Sqrt(dsum)

	thresha := math.Max(threshFactor*eps*dnorma, smlnum)
	threshb := math.Max(threshFactor*eps*dnormb, smlnum)

	if m == 2 {
		// Case 1: Swap 1×1 and 1×1 blocks.
		// Compute orthogonal QL and RQ that swap 1×1 and 1×1 blocks
		// using Givens rotations and perform swap tentatively.

		f = s[1*ldst+1]*t[0*ldst+0] - t[1*ldst+1]*s[0*ldst+0]
		g = s[1*ldst+1]*t[0*ldst+1] - t[1*ldst+1]*s[0*ldst+1]
		sa = math.Abs(s[1*ldst+1]) * math.Abs(t[0*ldst+0])
		sb = math.Abs(s[0*ldst+0]) * math.Abs(t[1*ldst+1])

		ir[1], ir[0], _ = impl.Dlartg(f, g)
		ir[1*ldst] = -ir[1]
		ir[1*ldst+1] = ir[0]

		bi.Drot(2, s[:], ldst, s[1:], ldst, ir[0], ir[1*ldst])
		bi.Drot(2, t[:], ldst, t[1:], ldst, ir[0], ir[1*ldst])

		if sa >= sb {
			li[0], li[1*ldst], _ = impl.Dlartg(s[0], s[1*ldst])
		} else {
			li[0], li[1*ldst], _ = impl.Dlartg(t[0], t[1*ldst])
		}
		bi.Drot(2, s[0:], 1, s[1*ldst:], 1, li[0], li[1*ldst])
		bi.Drot(2, t[0:], 1, t[1*ldst:], 1, li[0], li[1*ldst])
		li[1*ldst+1] = li[0]
		li[1] = -li[1*ldst]

		// Weak stability test: |s21| <= O(eps F-norm((A))); and |t21| <= O(eps F-norm((B)))

		weak = math.Abs(s[1*ldst]) <= thresha && math.Abs(t[1*ldst]) <= threshb
		if !weak {
			return true, false
		}
		if wands {
			// Strong stability test:
			//  f-norm((A-QLᴴ*S*QR)) <= O(eps*f-norm((A)))
			//  f-norm((B-QLᴴ*T*QR)) <= O(eps*f-norm((B)))
			impl.Dlacpy(blas.All, m, m, a[j1*lda+j1:], lda, work, ldw)
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li[:], ldst, s[:], ldst, 0, work, ldw)
			bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, -1, work, ldw, ir[:], ldst, 1, work[woffset:], ldw)
			dscale = 0
			dsum = 1
			dscale, dsum = impl.Dlassq(m*m, work[woffset:], ldw, dscale, dsum)
			sa = dscale * math.Sqrt(dsum)
			impl.Dlacpy(blas.All, m, m, b[j1*ldb+j1:], ldb, work[woffset:], ldw)
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li[:], ldst, t[:], ldst, 0, work, ldw)
			bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, -1, work, ldw, ir[:], ldst, 1, work[woffset:], ldw)
			dscale = 0
			dsum = 1
			dscale, dsum = impl.Dlassq(m*m, work[woffset:], ldw, dscale, dsum)
			sb = dscale * math.Sqrt(dsum)
			strong = sa <= thresha && sb <= threshb
			if !strong {
				return true, false
			}
		} // End wands if.

		// Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).

		bi.Drot(j1+1, a[j1:], lda, a[j1+1:], lda, ir[0], ir[1*ldst])
		bi.Drot(j1+1, b[j1:], ldb, b[j1+1:], ldb, ir[0], ir[1*ldst])
		bi.Drot(n-j1+1, a[j1*lda+j1:], 1, a[(j1+1)*lda+j1:], 1, li[0], li[1*ldst])
		bi.Drot(n-j1+1, b[j1*ldb+j1:], 1, b[(j1+1)*ldb+j1:], 1, li[0], li[1*ldst])

		// Set n1×n2 (2×1)  blocks to 0.
		a[(j1+1)*lda+j1] = 0
		b[(j1+1)*ldb+j1] = 0

		// Accumulate transformations into Q and Z if requested.

		if wantz {
			bi.Drot(n, z[j1:], ldz, z[j1+1:], ldz, ir[0], ir[1*ldst])
		}
		if wantq {
			bi.Drot(n, q[j1:], ldq, q[j1+1:], ldq, li[0], li[1*ldst])
		}
		return false, false // Succesful swap.
	}

	// Case 2: Swap 1×1 and 2×2 blocks or 2×2 and 2×2 blocks.

	// Solve for generalized sylvester equation:
	//  s11 * R - L * s22 = scale * s12
	//  t11 * R - L * t22 = scale * t12
	// for R and L. Solutions in li and ir.

	impl.Dlacpy(blas.All, n1, n2, t[n1:], ldst, li[:], ldst)
	impl.Dlacpy(blas.All, n1, n2, s[n1:], ldst, ir[n2*ldst+n1:], ldst)

	const noIjob = 0
	scale, dscale, dsum, _, linfo = impl.Dtgsy2(blas.NoTrans, noIjob, n1, n2, s[:], ldst, s[n1*ldst+n1:], ldst, ir[n2*ldst+n1:], ldst, t[:], ldst, t[n1*ldst+n1:], ldst, li[:], ldst, dsum, dscale, iwork[:])
	if linfo >= 0 {
		// Z perturbed to avoid underflow.
		return true, false
	}

	// Compute transposed orthogonal matrix QL:
	//  QLᵀ * li  = [TL; 0]
	// where
	//  li = [ -L; scale*identity(n2) ]

	for i := 0; i < n2; i++ {
		bi.Dscal(n1, -1, li[i:], ldst)
		li[(n1+i)*ldst+i] = scale
	}

	impl.Dgeqr2(m, n2, li[:], ldst, taul[:], work)
	impl.Dorg2r(m, m, n2, li[:], ldst, taul[:], work)

	// Compute orthogonal matrix RQ:
	//  ir * RQᵀ = [0 TR]
	// where ir = [scale*identity, R]

	for i := 0; i < n1; i++ {
		ir[(n2+i)*ldst+i] = scale
	}

	impl.Dgeqr2(n1, m, ir[(n2)*ldst:], ldst, taur[:], work)
	impl.Dorg2r(m, m, n1, ir[:], ldst, taur[:], work)

	// Perform swap tentatively.

	bi.Dgemm(blas.Trans, blas.NoTrans, m, m, m, 1, li[:], ldst, s[:], ldst, 0, work, m)
	bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, 1, work, m, ir[:], ldst, 0, s[:], ldst)
	bi.Dgemm(blas.Trans, blas.NoTrans, m, m, m, 1, li[:], ldst, t[:], ldst, 0, work, ldw)
	bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, 1, work, ldw, ir[:], ldst, 0, t[:], ldst)

	impl.Dlacpy(blas.All, m, m, s[:], ldst, scpy[:], ldst)
	impl.Dlacpy(blas.All, m, m, t[:], ldst, tcpy[:], ldst)
	impl.Dlacpy(blas.All, m, m, ir[:], ldst, ircop[:], ldst)
	impl.Dlacpy(blas.All, m, m, li[:], ldst, licop[:], ldst)

	// Triangularize the B-part by an RQ factorization.
	// Apply transformation (from left) to A-part, giving S.

	impl.Dgerq2(m, m, t[:], ldst, taur[:], work)
	impl.Dormr2(blas.Right, blas.Trans, m, m, m, t[:], ldst, taur[:], s[:], ldst, work)

	// Compute F-norm(S21) in brqa21. T21 is 0.
	dscale = 0
	dsum = 1
	for i := 0; i < n2; i++ {
		dscale, dsum = impl.Dlassq(n1, s[(n2)*ldst+i:], ldst, dscale, dsum)
	}
	bRqa21 = dscale * math.Sqrt(dsum)

	// Triangularize the B-part by a QR factorization.
	// Apply transformation (from right) to A-part, giving S.

	impl.Dgeqr2(m, m, tcpy[:], ldst, taul[:], work)
	impl.Dorm2r(blas.Left, blas.Trans, m, m, m, tcpy[:], ldst, taul[:], scpy[:], ldst, work)
	impl.Dorm2r(blas.Right, blas.NoTrans, m, m, m, tcpy[:], ldst, taul[:], licop[:], ldst, work)

	// Compute f-norm(s21) in bqra21. T21 is 0.

	dscale = 0
	dsum = 1
	for i := 0; i < n2; i++ {
		dscale, dsum = impl.Dlassq(n1, scpy[(n2)*ldst+i:], ldst, dscale, dsum)
	}
	bQra21 = dscale * math.Sqrt(dsum)

	// Decide which method to use.
	// Weak stability test:
	//  f-norm(s21) <= O(eps*f -norm((S)))

	if bQra21 <= bRqa21 && bQra21 <= thresha {
		impl.Dlacpy(blas.All, m, m, scpy[:], ldst, s[:], ldst)
		impl.Dlacpy(blas.All, m, m, tcpy[:], ldst, t[:], ldst)
		impl.Dlacpy(blas.All, m, m, ircop[:], ldst, ir[:], ldst)
		impl.Dlacpy(blas.All, m, m, licop[:], ldst, li[:], ldst)
	} else if bRqa21 >= thresha {
		return true, false
	}
	impl.Dlaset(blas.Lower, m-1, m-1, 0, 0, t[2*ldst:], ldst)

	if wands {
		// Strong stability test:
		// f-norm((A-QLᴴ*S*QR)) <= O(eps*f-norm((A))) and
		// f-norm((B-QLᴴ*T*QR)) <= O(eps*f-norm((B)))
		woffset := m * m
		impl.Dlacpy(blas.All, m, m, a[j1*lda+j1:], lda, work[woffset:], ldw)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li[:], ldst, s[:], ldst, 0, work, ldw)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, -1, work, ldw, ir[:], ldst, 1, work[woffset:], ldw)

		dscale = 0
		dsum = 1
		dscale, dsum = impl.Dlassq(m*m, work[woffset:], ldw, dscale, dsum)
		sa = dscale * math.Sqrt(dsum)

		impl.Dlacpy(blas.All, m, m, b[j1*ldb+j1:], ldb, work[woffset:], ldw)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li[:], ldst, t[:], ldst, 0, work, ldw)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, -1, work, ldw, ir[:], ldst, 1, work[woffset:], ldw)

		dscale = 0
		dsum = 1
		dscale, dsum = impl.Dlassq(m*m, work[woffset:], ldw, dscale, dsum)
		sb = dscale * math.Sqrt(dsum)
		if sa <= thresha && sb <= threshb {
			return true, false
		}
	}

	// If swap accepted, apply transformations and set n1×n2 (2×1) blocks to 0.

	impl.Dlaset(blas.All, n1, n2, 0, 0, s[(n2)*ldst:], ldst)

	// Copy back m×m diagonal block starting at index j1 of (A,B).
	impl.Dlacpy(blas.All, m, m, s[:], ldst, a[j1*lda+j1:], lda)
	impl.Dlacpy(blas.All, m, m, t[:], ldst, b[j1*ldb+j1:], ldb)

	impl.Dlaset(blas.All, ldst, ldst, 0, 0, t[:], ldst)

	// Standardize existing 2×2 blocks.
	impl.Dlaset(blas.All, m, m, 0, 0, work, ldw)
	// TODO work[:m*m] is a vector that represents a m×m matrix!
	// Revise all indexing below.
	work[0] = 1
	t[0] = 1
	if n2 > 1 {
		// Missing Dlagv2
		work[ldw] = -work[1]
		work[ldw+1] = work[0]
		t[(n2)*ldst+n2] = t[0]
		t[1] = -t[ldst]
	}
	work[woffset-1] = 1
	t[m*ldst+m] = 1

	if n1 > 1 {
		// Missing Dlagv2
		work[woffset-1] = work[(n2-1)*ldw+n2]
		work[woffset-2] = -work[(n2-1)*ldw+n2+1]
		t[m*ldst+m] = t[(n2)*ldst+n2]
		t[(m-1)*ldst+m] = -t[m*ldst+m-1]
	}

	bi.Dgemm(blas.Trans, blas.NoTrans, n2, n1, n2, 1, work, ldw, a[j1*lda+j1+n2-1:], lda, 0, work[woffset:], n2)
	impl.Dlacpy(blas.All, n2, n1, work[woffset:], n2, a[j1*lda+j1+n2-1:], lda)

	bi.Dgemm(blas.Trans, blas.NoTrans, n2, n1, n2, 1, work, ldw, b[j1*ldb+j1+n2-1:], ldb, 0, work[woffset:], n2)
	impl.Dlacpy(blas.All, n2, n1, work[woffset:], n2, b[j1*ldb+j1+n2-1:], ldb)

	bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li[:], ldst, work, ldw, 0, work[woffset:], ldw)
	impl.Dlacpy(blas.All, m, m, work[woffset:], ldw, li[:], ldst)

	bi.Dgemm(blas.NoTrans, blas.NoTrans, n2, n1, n1, 1, a[j1*lda+j1+n2-1:], lda, t[(n2)*ldst+n2:], ldst, 0, work, n2)
	impl.Dlacpy(blas.All, n2, n1, work, n2, a[j1*lda+j1+n2-1:], lda)

	bi.Dgemm(blas.NoTrans, blas.NoTrans, n2, n1, n1, 1, b[j1*ldb+j1+n2-1:], ldb, t[(n2)*ldst+n2:], ldst, 0, work, n2)
	impl.Dlacpy(blas.All, n2, n1, work, n2, b[j1*ldb+j1+n2-1:], ldb)

	bi.Dgemm(blas.Trans, blas.NoTrans, m, m, m, 1, ir[:], ldst, t[:], ldst, 0, work, ldw)
	impl.Dlacpy(blas.All, m, m, work, ldw, ir[:], ldst)

	// Accumulate transformation into Q and Z if requested.
	if wantq {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, q[j1:], ldq, li[:], ldst, 0, work, n)
		impl.Dlacpy(blas.All, n, m, work, n, q[j1:], ldq)
	}
	if wantz {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, z[j1:], ldz, ir[:], ldst, 0, work, n)
		impl.Dlacpy(blas.All, n, m, work, n, z[j1:], ldz)
	}

	// Update (A(J1:J1+M-1, M+J1:N), B(J1:J1+M-1, M+J1:N)) and (A(1:J1-1, J1:J1+M), B(1:J1-1, J1:J1+M)).

	i := j1 + m - 1
	if i < n {
		bi.Dgemm(blas.Trans, blas.NoTrans, m, n-i, m, 1, li[:], ldst, a[j1*ldst+i:], lda, 0, work, ldw)
		impl.Dlacpy(blas.All, m, n-i, work, ldw, a[j1*lda+i:], lda)

		bi.Dgemm(blas.Trans, blas.NoTrans, m, n-i, m, 1, li[:], ldst, b[j1*ldb+i:], ldb, 0, work, ldw)
		impl.Dlacpy(blas.All, m, n-i, work, ldw, b[j1*ldb+i:], ldb)
	}

	i = j1 - 1
	if i >= 0 {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, i+1, m, m, 1, a[j1:], lda, ir[:], ldst, 0, work, i+1)
		impl.Dlacpy(blas.All, i+1, m, work, i+1, a[j1:], lda)

		bi.Dgemm(blas.NoTrans, blas.NoTrans, i+1, m, m, 1, b[j1:], ldb, ir[:], ldst, 0, work, i+1)
		impl.Dlacpy(blas.All, i+1, m, work, i+1, b[j1:], ldb)
	}

	// Conditioning OK, swap successful.
	return false, false
}
