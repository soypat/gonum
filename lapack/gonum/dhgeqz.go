// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
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
		iiter                  int
		ilschr, ilq, ilz       bool
		ischur, icompq, icompz int
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
	impl.Dlanhs(lapack.MaxColumnSum, in, h[ilo*ldh+ilo:], ldh, work)

	return info
}
