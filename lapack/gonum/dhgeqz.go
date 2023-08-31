// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dhgeqz computes the eigenvalues of a real matrix pair (H,T),
// where H is an upper Hessenberg matrix and T is upper triangular,
// using the double-shift QZ method.
// Matrix pairs of this type are produced by the reduction to
// generalized upper Hessenberg form of a real matrix pair (A,B):
//
//	A = Q1*H*Z1ᵀ,  B = Q1*T*Z1ᵀ,
//
// as computed by DGGHRD.
// If JOB='S', then the Hessenberg-triangular pair (H,T) is
// also reduced to generalized Schur form,
//
//	H = Q*S*Zᵀ,  T = Q*P*Zᵀ,
//
// where Q and Z are orthogonal matrices, P is an upper triangular
// matrix, and S is a quasi-triangular matrix with 1-by-1 and 2-by-2
// diagonal blocks.
// The 1-by-1 blocks correspond to real eigenvalues of the matrix pair
// (H,T) and the 2-by-2 blocks correspond to complex conjugate pairs of
// eigenvalues.
// Additionally, the 2-by-2 upper triangular diagonal blocks of P
// corresponding to 2-by-2 blocks of S are reduced to positive diagonal
// form, i.e., if S(j+1,j) is non-zero, then P(j+1,j) = P(j,j+1) = 0,
// P(j,j) > 0, and P(j+1,j+1) > 0.
//
// Optionally, the orthogonal matrix Q from the generalized Schur
// factorization may be postmultiplied into an input matrix Q1, and the
// orthogonal matrix Z may be postmultiplied into an input matrix Z1.
// If Q1 and Z1 are the orthogonal matrices from DGGHRD that reduced
// the matrix pair (A,B) to generalized upper Hessenberg form, then the
// output matrices Q1*Q and Z1*Z are the orthogonal factors from the
// generalized Schur factorization of (A,B):
//
//	A = (Q1*Q)*S*(Z1*Z)ᵀ,  B = (Q1*Q)*P*(Z1*Z)ᵀ.
//
// To avoid overflow, eigenvalues of the matrix pair (H,T) (equivalently,
// of (A,B)) are computed as a pair of values (alpha,beta), where alpha is
// complex and beta real.
// If beta is nonzero, lambda = alpha / beta is an eigenvalue of the
// generalized nonsymmetric eigenvalue problem (GNEP)
//
//	A*x = lambda*B*x
//
// and if alpha is nonzero, mu = beta / alpha is an eigenvalue of the
// alternate form of the GNEP
//
//	mu*A*y = B*y.
//
// Real eigenvalues can be read directly from the generalized Schur
// form:
//
//	alpha = S(i,i), beta = P(i,i).
//
// - info=-1: successful exit
// - info>=0:
//   - info<=n: The QZ iteration did not converge. (H,T) is not in Schur form
//     but alphar(i), alphai(i), and beta(i), i=info+1,...,n should be correct.
//   - info=n+1...2*n: The shift calculation failed. (H,T) is not in Schur form
//     but alphar(i), alphai(i), and beta(i), i=info-n+1,...,n should be correct.
//
// Ref: C.B. Moler & G.W. Stewart, "An Algorithm for Generalized Matrix Eigenvalue Problems", SIAM J. Numer. Anal., 10(1973), pp. 241--256.
// https://doi.org/10.1137/0710024
func (impl Implementation) Dhgeqz(job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi int, h []float64, ldh int, t []float64, ldt int, alphar, alphai, beta, q []float64, ldq int, z []float64, ldz int, work []float64, workspaceQuery bool) (info int) {
	var (
		jiter int // counts iterations
		// counts iterations run since ILAST was last changed.
		//This is therefore reset only when a 1-by-1 or  2-by-2 block deflates off the bottom.
		iiter                                                       int
		ilschr, ilq, ilz, ilazro, ilazr2                            bool
		ischur, icompq, icompz, ifirst, istart                      int
		c, s, s1, s2, wr, wr2, wi, scale, temp, tempr, temp2, tempi float64 // Trigonometric temporary variables.
		b22, b11, sr, cr, sl, cl, cz, t1, slinv, szr, szi           float64
		a11, a21, a12, a22, c11r, c11i, c12, c21, c22r, c22i        float64
	)
	switch job {
	case lapack.EigenvaluesOnly:
		ilschr = false
		ischur = 1
	case lapack.EigenvaluesAndSchur:
		ilschr = true
		ischur = 2
	default:
		panic(badSchurJob)
	}

	switch compq {
	case lapack.SchurNone:
		ilq = false
		icompq = 1
	case lapack.SchurOrig:
		ilq = true
		icompq = 2
	case lapack.SchurHess:
		ilq = true
		icompq = 3
	default:
		panic(badSchurComp)
	}
	switch compz {
	case lapack.SchurNone:
		ilz = false
		icompz = 1
	case lapack.SchurOrig:
		ilz = true
		icompz = 2
	case lapack.SchurHess:
		ilz = true
		icompz = 3
	default:
		panic(badSchurComp)
	}
	lwork := len(work)
	switch {
	case n < 0:
		panic(nLT0)
	case ilo < 0:
		panic(badIlo)
	case ilo-1 < ihi || ihi > n:
		panic(badIhi)
	case ldh < n:
		panic(badLdH)
	case ldt < n:
		panic(badLdT)
	case ldq < 1 || (ilq && ldq < n):
		panic(badLdQ)
	case ldz < 1 || (ilz && ldz < n):
		panic(badLdZ)
	case lwork < max(1, n) && !workspaceQuery:
		panic(badLWork)
	}
	info = -1 // info==-1 is succesful exit.
	if n == 0 {
		work[0] = 1
		return info
	}

	// Initialize Q and Z.
	if icompq == 3 {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}
	if icompz == 3 {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz)
	}

	// Machine constants.
	const (
		safmin = dlamchS
		safmax = 1. / safmin
		ulp    = dlamchE * dlamchB
	)

	in := ihi + 1 - ilo
	bi := blas64.Implementation()
	anorm := impl.Dlange(lapack.MaxColumnSum, in, in, h[ilo*ldh+ilo:], ldh, work)
	bnorm := impl.Dlange(lapack.MaxColumnSum, in, in, t[ilo*ldt+ilo:], ldt, work)

	atol := math.Max(safmin, ulp*anorm)
	btol := math.Max(safmin, ulp*bnorm)
	ascale := 1. / math.Max(safmin, anorm)
	bscale := 1. / math.Max(safmin, bnorm)

	// Set eigenvalues ihi+1 to n.
	for j := ihi + 1; j < n; j++ {
		if t[j*ldt+j] < 0 {
			if ilschr {
				for jr := 0; jr < j; jr++ {
					h[jr*ldh+j] = -h[jr*ldh+j]
					t[jr*ldt+j] = -t[jr*ldt+j]
				}
			} else {
				h[j*ldh+j] = -h[j*ldh+j]
				t[j*ldt+j] = -t[j*ldt+j]
			}
			if ilz {
				for jr := 0; jr < n; jr++ {
					z[jr*ldz+j] = -z[jr*ldz+j]
				}
			}
		}
		alphar[j] = h[j*ldh+j]
		alphai[j] = 0
		beta[j] = t[j*ldt+j]
	}

	// If ihi<ilo, skip QZ steps.
	if ihi < ilo {
		panic('L')
		// goto 380
	}

	// MAIN QZ ITERATION LOOP
	// Initialize dynamic indices.
	// Eigenvalues ILAST+1:N have been found.
	ilast := ihi
	ifrstm := ilo
	ilastm := ihi
	if ilschr {
		ifrstm = 1
		ilastm = n
	}

	eshift := 0.0
	maxiter := 30 * (ihi - ilo + 1)
	for jiter = 1; jiter <= maxiter; jiter++ {
		// Split the matrix if possible.
		// Two tests:
		//  1: H(j,j-1)=0  or  j=ILO
		//  2: T(j,j)=0
		if ilast == ilo {
			// special case: j=ilast
			goto Eighty
		}
		if math.Abs(h[ilast*ldh+ilast-1]) <= math.Max(safmin, ulp*(math.Abs(h[ilast*ldh+ilast])+math.Abs(h[(ilast-1)*ldh+ilast-1]))) {
			h[ilast*ldh+ilast-1] = 0
			goto Eighty
		}
		if math.Abs(t[ilast*ldt+ilast]) <= btol {
			t[ilast*ldt+ilast] = 0
			goto Seventy
		}

		// General case: j<ilast.

		for j := ilast - 1; j >= ilo; j-- {
			// Test 1: for H(j,j-1)=0 or j=ILO
			if j == ilo {
				ilazro = true
			} else {
				if math.Abs(h[j*ldh+j-1]) <= math.Max(safmin, ulp*(math.Abs(h[j*ldh+j])+math.Abs(h[(j-1)*ldh+j-1]))) {
					h[j*ldh+j-1] = 0
					ilazro = true
				} else {
					ilazro = false
				}
			}

			// Test 2: for T(j,j)=0

			if math.Abs(t[j*ldt+j]) < btol {
				t[j*ldt+j] = 0
				//Test 1a: Check for 2 consecutive small subdiagonals in A.
				ilazr2 = false
				if !ilazro {
					temp = math.Abs(h[j*ldh+j-1]) + math.Abs(h[(j-1)*ldh+j-2])
					temp2 := math.Abs(h[j*ldh+j])
					tempr = math.Max(temp, temp2)
					if tempr < 1 && tempr != 0 {
						temp /= tempr
						temp2 /= tempr
					}
					if temp*(ascale*math.Abs(h[(j+1)*ldh+j])) <= temp2*(ascale*atol) {
						ilazr2 = true
					}
				}
				// If both tests pass (1 & 2), i.e., the leading diagonal
				// element of B in the block is zero, split a 1x1 block off
				// at the top. (I.e., at the J-th row/column) The leading
				// diagonal element of the remainder can also be zero, so
				// this may have to be done repeatedly.
				if ilazro || ilazr2 {
					for jch := j; jch <= ilast-1; jch++ {
						temp := h[jch*ldh+jch]
						c, s, h[jch*ldh+jch] = impl.Dlartg(temp, h[(jch+1)*ldh+jch])
						t[(jch+1)*ldt+jch+1] = 0
						if jch < ilastm-1 {
							bi.Drot(ilastm-jch-1, t[jch*ldt+jch+2:], 1, t[(jch+1)*ldt+jch+2:], 1, c, s)
						}
						bi.Drot(ilastm-jch+2, h[jch*ldh+jch-1:], 1, h[(jch+1)*ldh+jch-1:], 1, c, s)

						if ilq {
							bi.Drot(n, q[jch:], ldq, q[jch+1:], ldq, c, s)
						}
						temp = h[(jch+1)*ldh+jch]
						c, s, h[(jch+1)*ldh+jch] = impl.Dlartg(temp, h[(jch+1)*ldh+jch-1])
						h[(jch+1)*ldh+jch-1] = 0
						bi.Drot(jch+1-ifrstm, h[ifrstm*ldh+jch:], ldh, h[ifrstm*ldh+jch-1:], ldh, c, s)
						bi.Drot(jch-ifrstm, t[ifrstm*ldt+jch:], ldt, t[ifrstm*ldt+jch-1:], ldt, c, s)
						if ilz {
							bi.Drot(n, z[jch:], ldz, z[jch-1:], ldz, c, s)
						}
					}
				}
			} else if ilazro {
				// Only test 1 passed -- work on j:ilast.
				ifirst = j
				// goto 110
			}
			// Neither test passed -- try next j.
		}
	}
	panic("unreachable")

Seventy:
	// T(ILAST,ILAST)=0 -- clear H(ILAST,ILAST-1) to split off a
	// 1x1 block.
	temp = h[ilast*ldh+ilast-1]
	c, s, h[ilast*ldh+ilast] = impl.Dlartg(temp, h[ilast*ldh+ilast-1])
	h[ilast*ldh+ilast-1] = 0
	bi.Drot(ilast-ifrstm, h[ifrstm*ldh+ilast:], ldh, h[ifrstm*ldh+ilast-1:], ldh, c, s)
	bi.Drot(ilast-ifrstm, t[ifrstm*ldt+ilast:], ldt, t[ifrstm*ldt+ilast-1:], ldt, c, s)
	if ilz {
		bi.Drot(n, z[ilast:], ldz, z[ilast-1:], ldz, c, s)
	}

	// H(ILAST,ILAST-1)=0 -- Standardize B, set ALPHAR, ALPHAI, and BETA.

Eighty:
	if t[ilast*ldt+ilast] < 0 {
		if ilschr {
			for j := ifrstm; j <= ilast; j++ {
				h[j*ldh+j] = -h[j*ldh+j]
				t[j*ldt+j] = -t[j*ldt+j]
			}
		} else {
			h[ilast*ldh+ilast] = -h[ilast*ldh+ilast]
			t[ilast*ldt+ilast] = -t[ilast*ldt+ilast]
		}
		if ilz {
			for j := 0; j < n; j++ {
				z[j*ldz+ilast] = -z[j*ldz+ilast]
			}
		}
	}
	alphar[ilast] = h[ilast*ldh+ilast]
	alphai[ilast] = 0
	beta[ilast] = t[ilast*ldt+ilast]

	// Go to next block -- exit if finished.
	ilast--
	if ilast < ilo {
		// goto 380
	}
	// Reset counters.

	iiter = 0
	eshift = 0.0
	if !ilschr {
		ilastm = ilast
		if ifrstm > ilast {
			ifrstm = ilo
		}
	}
	// goto 350.

	// QZ Step
	// This iteration only involves rows/columns IFIRST:ILAST. We
	// assume IFIRST < ILAST, and that the diagonal of B is non-zero.
	goto OneTen
OneTen:
	iiter++
	if !ilschr {
		ifrstm = ifirst
	}

	// Compute single shifts.
	// At this point, IFIRST < ILAST, and the diagonal elements of
	// T(IFIRST:ILAST,IFIRST,ILAST) are larger than BTOL (in magnitude).

	if (iiter/10)*10 == iiter {
		// Exceptional shift. Chosen for no particularly good reason. (single shift only)
		if float64(maxiter)*safmin*math.Abs(h[ilast*ldh+ilast-1]) < math.Abs(t[(ilast-1)*ldt+ilast-1]) {
			eshift = h[ilast*ldh+ilast-1] / t[(ilast-1)*ldt+ilast-1]
		} else {
			eshift += 1 / (safmin * float64(maxiter))
		}
		s1 = 1
		wr = eshift
	} else {
		// Shifts based on the generalized eigenvalues of the
		// bottom-right 2x2 block of A and B. The first eignevalue
		// returned by Dlag2 is the wilkinson shift (AEP p.512).
		s1, s2, wr, wr2, wi = impl.Dlag2(h[(ilast-1)*ldh+ilast-1:], ldh, t[(ilast-1)*ldt+ilast-1:], ldt)
		hlast := h[ilast*ldh+ilast]
		tlast := t[ilast*ldt+ilast]
		if math.Abs((wr/s1)*tlast-hlast) > math.Abs((wr2/s2)*tlast-hlast) {
			wr, wr2 = wr2, wr
			s1, s2 = s2, s1
		}
		temp = math.Max(s1, safmin*math.Max(1, math.Max(math.Abs(wr), math.Abs(wi))))
		if wi != 0 {
			goto TwoHundred
		}
	}

	// Fiddle with shift to avoid overflow.
	temp = math.Min(ascale, 1) * (safmax / 2)
	if s1 > temp {
		scale = temp / s1
	} else {
		scale = 1
	}

	temp = math.Min(bscale, 1) * (safmax / 2)
	if math.Abs(wr) > temp {
		scale = math.Min(scale, temp/math.Abs(wr))
	}
	s1 *= scale
	wr *= scale

	// Now check for two consecutive small subdiagonals.
	for j := ilast - 1; j >= ifirst+1; j-- {
		istart = j
		temp = math.Abs(s1 * h[j*ldh+j-1])
		temp2 := math.Abs(s1*h[j*ldh+j] - wr*t[j*ldt+j])
		tempr = math.Max(temp, temp2)
		if tempr < 1 && tempr != 0 {
			temp /= tempr
			temp2 /= tempr
		}
		if math.Abs(ascale*h[(j+1)*ldh+j]*temp) <= ascale*atol*temp2 {
			goto OneThirty
		}
	}
	istart = ifirst

OneThirty:

	// Do an implicit-shift QZ sweep.
	// Initial Q

	temp = s1*h[istart*ldh+istart] - wr*t[istart*ldt+istart]
	c, s, tempr = impl.Dlartg(temp, s1*h[(istart+1)*ldh+istart])

	// Sweep.
	for j := istart; j < ilast-1; j++ {
		if j > istart {
			temp = h[j*ldh+j-1]
			c, s, h[j*ldh+j-1] = impl.Dlartg(temp, h[(j+1)*ldh+j-1])
			h[(j+1)*ldh+j-1] = 0
		}
		for jc := j; jc <= ilastm; jc++ {
			temp = c*h[j*ldh+jc] + s*h[(j+1)*ldh+jc]
			h[(j+1)*ldh+jc] = -s*h[j*ldh+jc] + c*h[(j+1)*ldh+jc]
			h[j*ldh+jc] = temp
			temp2 := c*t[j*ldt+jc] + s*t[(j+1)*ldt+jc]
			t[(j+1)*ldt+jc] = -s*t[j*ldt+jc] + c*t[(j+1)*ldt+jc]
			t[j*ldt+jc] = temp2
		}
		if ilq {
			for jr := 0; jr < n; jr++ {
				temp = c*q[jr*ldq+j] + s*q[jr*ldq+j+1]
				q[jr*ldq+j+1] = -s*q[jr*ldq+j] + c*q[jr*ldq+j+1]
				q[jr*ldq+j] = temp
			}
		}
		temp = t[(j+1)*ldt+j+1]
		c, s, t[(j+1)*ldt+j+1] = impl.Dlartg(temp, t[(j+1)*ldt+j])
		t[(j+1)*ldt+j] = 0
		maxjr := min(j+2, ilast)
		for jr := ifrstm; jr <= maxjr; jr++ {
			temp = c*h[jr*ldh+j+1] + s*h[jr*ldh+j]
			h[jr*ldh+j] = -s*h[jr*ldh+j+1] + c*h[jr*ldh+j]
			h[jr*ldh+j+1] = temp
		}
		for jr := ifrstm; jr <= ilastm; jr++ {
			temp = c*t[jr*ldt+j+1] + s*t[jr*ldt+j]
			t[jr*ldt+j] = -s*t[jr*ldt+j+1] + c*t[jr*ldt+j]
			t[jr*ldt+j+1] = temp
		}
		if ilz {
			for jr := 0; jr < n; jr++ {
				temp = c*z[jr*ldz+j+1] + s*z[jr*ldz+j]
				z[jr*ldz+j] = -s*z[jr*ldz+j+1] + c*z[jr*ldz+j]
				z[jr*ldz+j+1] = temp
			}
		}
	}
	// goto 350.

	// Use Francis double-shift.

TwoHundred:
	if ifirst+1 == ilast {
		// Special case -- 2x2 block with complex eigenvectors.
		// Step 1: Standardize, that is, rotate so that
		// B =  (B11  0 )
		//      ( 0  B22)   With B11 non-negative.
		b22, b11, sr, cr, sl, cl = impl.Dlasv2(t[(ilast-1)*ldt+ilast-1], t[(ilast-1)*ldt+ilast], t[ilast*ldt+ilast])
		if b11 < 0 {
			cr = -cr
			sr = -sr
			b11 = -b11
			b22 = -b22
		}
		bi.Drot(ilastm+1-ifirst, h[(ilast-1)*ldh+ilast-1:], 1, h[ilast*ldh+ilast-1:], 1, cl, sl)
		bi.Drot(ilast+1-ifrstm, h[ifrstm*ldh+ilast-1:], ldh, h[ifrstm*ldh+ilast:], ldh, cr, sr)

		if ilast < ilastm {
			bi.Drot(ilastm-ilast, t[(ilast-1)*ldt+ilast+1:], 1, t[ilast*ldt+ilast+1:], 1, cl, sl)
		}
		if ifrstm < ilast-1 {
			bi.Drot(ifirst-ifrstm, t[ifrstm*ldt+ilast-1:], ldt, t[ifrstm*ldt+ilast:], ldt, cr, sr)
		}

		if ilq {
			bi.Drot(n, q[ilast-1:], ldq, q[ilast:], ldq, cl, sl)
		}
		if ilz {
			bi.Drot(n, z[ilast-1:], ldz, z[ilast:], ldz, cr, sr)
		}

		t[(ilast-1)*ldt+ilast-1] = b11
		t[(ilast-1)*ldt+ilast] = 0
		t[ilast*ldt+ilast-1] = 0
		t[ilast*ldt+ilast] = b22

		// If B22 is negative, negate column ilast.
		if b22 < 0 {
			for j := ifrstm; j <= ilast; j++ {
				h[j*ldh+ilast] = -h[j*ldh+ilast]
				t[j*ldt+ilast] = -t[j*ldt+ilast]
			}
			if ilz {
				for j := 0; j < n; j++ {
					z[j*ldz+ilast] = -z[j*ldz+ilast]
				}
			}
			b22 = -b22
		}

		// Step 2: compute alphar, alphai, and beta.
		// Recompute shift.
		s1, temp, wr, temp2, wi = impl.Dlag2(h[(ilast-1)*ldh+ilast-1:], ldh, t[(ilast-1)*ldt+ilast-1:], ldt)
		if wi == 0 {
			// If standardization has perturbed the shift onto real line, do another QR step.
			// goto 350
		}
		slinv = 1 / s1
		// Do EISPACK (QZVAL) computation of alpha and beta.

		a11 = h[(ilast-1)*ldh+ilast-1]
		a21 = h[ilast*ldh+ilast-1]
		a12 = h[(ilast-1)*ldh+ilast]
		a22 = h[ilast*ldh+ilast]

		// Compute complex Givens rotation on right assuming some element of C = (sA -wB)>unfl:
		// ( sA - wB ) (  CZ    -SZ  )
		//             (  SZ     CZ  )

		c11r = s1*a11 - wr*b11
		c11i = -wi * b11
		c12 = s1 * a12
		c21 = s1 * a21
		c22r = s1*a22 - wr*b22
		c22i = -wi * b22

		if math.Abs(c11r)+math.Abs(c11i)+math.Abs(c12) > math.Abs(c21)+math.Abs(c22r)+math.Abs(c22i) {
			t1 = impl.Dlapy3(c12, c11r, c11i)
			cz = c12 / t1
			szr = -c11r / t1
			szi = -c11i / t1
		} else {
			cz = impl.Dlapy2(c22r, c22i)
			if cz <= safmin {
				cz = 0
				szr = 1
				szi = 0
			} else {
				tempr = c22r / cz
				tempi = c22i / cz
				t1 = impl.Dlapy2(cz, c21)
				cz = cz / t1
				szr = -c21 * tempr / t1
				szi = c21 * tempi / t1
			}
		}

		// Compute Givens rotation on left
		// ( CQ   SQ )
		// (-SQ   CQ )   A or B.

		// an =
	} // You are still not done here...
	return info
}
