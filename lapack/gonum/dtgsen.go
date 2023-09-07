// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
)

// Dtgsen reorders the generalized real Schur decomposition of a real
// matrix pair (A, B) (in terms of an orthonormal equivalence trans-
// formation Qᵀ * (A, B) * Z), so that a selected cluster of eigenvalues
// appears in the leading diagonal blocks of the upper quasi-triangular
// matrix A and the upper triangular B. The leading columns of Q and
// Z form orthonormal bases of the corresponding left and right eigen-
// spaces (deflating subspaces). (A, B) must be in generalized real
// Schur canonical form (as returned by DGGES), i.e. A is block upper
// triangular with 1×1 and 2×2 diagonal blocks. B is upper
// triangular.
//
// Dtgsen also computes the generalized eigenvalues
//
//	w(j) = (alphar(j) + i*alphai(j)) / beta(j)
//
// of the reordered matrix pair (A, B).
// Optionally, Dtgsen computes the estimates of reciprocal condition
// numbers for eigenvalues and eigenspaces. These are Difu[(A11,B11),
// (A22,B22)] and Difl[(A11,B11), (A22,B22)], i.e. the separation(s)
// between the matrix pairs (A11, B11) and (A22,B22) that correspond to
// the selected cluster and the eigenvalues outside the cluster, resp.,
// and norms of "projections" onto left and right eigenspaces w.r.t.
// the selected cluster in the (1,1)-block.
//
// Parameters:
//   - ijob specifies whether condition numbers are required for the
//     cluster of eigenvalues (PlasmaIvec) or the deflating subspaces (Difu and Difl).
//   - wantq, wantz indicate whether to update the left and/or right transformation matrices.
//     q or z and not referenced if wantq or wantz are false, respectively.
//   - bselect is a boolean array of dimension n which specifies the eigenvalues in the selected cluster.
//     To select a real eigenvalue w[j], bselect[j] must be set to true.
//     To select a complex conjugate pair of eigenvalues w[j] and w[j+1], corresponding to a 2×2 diagonal block,
//     either bselect[j] or bselect[j+1] or both must be set to true. A complex conj. pair of EV must be either both
//     included in the cluster or both excluded.
//   - n is order of matrices A and B. n>=0.
//   - lda and ldb are leading dimensions of A and B, respectively.
//   - a and b contain the matrices A and B, respectively, in generalized real Schur canonical form.
//   - alphar, alphai and beta are arrays of length n which will contain the
//     generalized eigenvalues on Dtgsen's exit.
//   - m is the dimension of the specified pair of left and right eigenspaces. 0<=m<=n.
//   - q and z contain n×n matrices on entry and on exit have been postmultiplied by the
//     left orthogonal transformation matrix which reorder (A,B); the leading m columns
//     of these matrices form orthonormal basesfor the specified pair of left eigenspaces (deflating subspaces).
//   - pl, pr are lower bounds on the reciprocal of the norm of "projections" onto left and right eigenspaces
//     with respect to the selected cluster. pl>0, pr<=1. If ijob is not 1,4,5 pl and pr are undefined.
//   - dif stores estimates of Difl and Difu on ijob>=2. If ijob is 2 or 4 dif stores F-norm-base upper bounds.
//     If ijob is 3 or 5 dif stores 1-norm-based estimates of Difu and Difl.
//   - work is of length max(1, lwork). On workspace query and on successful exit work[0] contains optimal lwork.
//   - iwork is of length max(1, liwork). On workspace query and on successful exit iwork[0] contains optimal liwork.
//   - illConditioned is true if the matrix pair (A,B) has been detected to be ill-conditioned
func (impl Implementation) Dtgsen(ijob int, wantq, wantz bool, bselect []bool, n int, a []float64, lda int, b []float64, ldb int, alphar, alphai, beta, q []float64, ldq int, z []float64, ldz int, work []float64, iwork []int, isWorkspaceQuery bool) (pl, pr float64, dif [2]float64, m int, illConditioned, lworkTooSmall, liworkTooSmall bool) {
	// Early check for workspace query calculation.
	switch {
	case n < 0:
		panic(nLT0)
	case m < 0:
		panic(mLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case len(work) < 1:
		panic(shortWork)
	case len(iwork) < 1:
		panic(shortIWork)
	}

	const (
		eps    = dlamchP
		smlnum = dlamchS / eps
	)

	// Set m to the dimension of the specified pair of deflating subspaces.
	m = 0
	pair := false
	if !isWorkspaceQuery || ijob != 0 {
		for k := 0; k < n; k++ {
			if pair {
				pair = false
			} else {
				if k < n-1 {
					if a[(k+1)*lda+k] == 0 && bselect[k] {
						m++
					} else {
						pair = true
						if bselect[k] || bselect[k+1] {
							m += 2
						}
					}
				} else {
					if bselect[n-1] {
						m++
					}
				}
			}
		}
	}

	// Calculate workspace necessary.
	var lwmin, liwmin int
	switch ijob {
	case 0:
		lwmin = max(1, 4*n+16)
		liwmin = 1
	case 1, 2, 4:
		lwmin = max(1, max(4*n+16, 2*m*(n-m)))
		liwmin = max(1, n+6)
	case 3, 5:
		lwmin = max(1, max(4*n+16, 4*m*(n-m)))
		liwmin = max(1, max(2*m*(n-m), n+6))
	default:
		panic(badIJob)
	}

	if isWorkspaceQuery {
		work[0] = float64(lwmin)
		iwork[0] = liwmin
		return pl, pr, dif, m, false, false, false
	}

	// Second check for rest of parameters.
	switch {
	case len(work) < lwmin:
		panic(shortWork)
	case len(iwork) < liwmin:
		panic(shortIWork)
	case ldq < 1 || wantq && ldq < n:
		panic(badLdQ)
	case ldz < 1 || wantz && ldz < n:
		panic(badLdZ)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(alphar) < n:
		panic(badLenAlpha)
	case len(alphai) < n:
		panic(badLenAlpha)
	case len(beta) < n:
		panic(badLenBeta)
	case len(bselect) != n:
		panic(shortSelect)
	}

	wantp := ijob == 1 || ijob == 4 || ijob == 5
	wantd1 := ijob == 2 || ijob == 4 || ijob == 5
	wantd2 := ijob == 3 || ijob == 5
	wantd := wantd1 || wantd2

	idifjob := 3

	if m == n || m == 0 {
		// Quicker return if possible.
		if wantp {
			pl = 1
			pr = 1
		}
		if wantd {
			dscale := 0.0
			dsum := 1.0
			for i := 0; i < n; i++ {
				dscale, dsum = impl.Dlassq(n, a[i:], lda, dscale, dsum)
				dscale, dsum = impl.Dlassq(n, b[i:], ldb, dscale, dsum)
			}
			dif[0] = dscale * math.Sqrt(dsum)
			dif[1] = dif[0]
		}
		goto Sixty
	}

	// Collect the selected blocks at the top-left corner of (A,B).

	ks := 0
	pair = false
	for k := 0; k < n; k++ {
		if pair {
			pair = false
		} else {
			swap := bselect[k]
			if k < n-1 && a[(k+1)*lda+k] != 0 {
				pair = true
				swap = swap || bselect[k]
			}
			if swap {
				ks++
				// Swap the K-th block to position KS.
				// Perform reordering of diagonal blocks in (A,B)
				// by orthogonal transformation matrices and update Q and Z
				// accordingly (if requested).

				kk := k
				if k != ks {
					kk, ks, illConditioned, lworkTooSmall = impl.Dtgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, kk, ks, work, false)
					if illConditioned {
						// Swap rejected.
						if wantp {
							pl, pr = 0, 0
						}
						if wantd {
							dif = [2]float64{}
						}
						goto Sixty
					}
					if pair {
						ks++
					}
				}
			}
		}
	} // 30 Continue.
	var (
		dscale float64
		ierr   int
		isave  [3]int
	)
	if wantp {
		// Solve generalized Sylvester equation for R and L
		// and compute PL and PR.
		n1 := m
		n2 := n - m
		i := n1
		ijb := 0
		ldw := n1
		impl.Dlacpy(blas.All, n1, n2, a[i:], lda, work, ldw)
		impl.Dlacpy(blas.All, n1, n2, b[i:], ldb, work[n1*n2:], ldw)

		dif[0], dscale, ierr = impl.Dtgsyl(blas.NoTrans, ijb, n1, n2, a, lda, a[i*lda+i:], lda, work, ldw, b, ldb, b[i*lda+i:], ldb, work[n1*n2:], ldw, work[n1*n2*2:], iwork, false)

		// Estimater the reciprocal of norms of projections onto left and right eigenspaces.
		rdscal := 0.0
		dsum := 1.0
		rdscal, dsum = impl.Dlassq(n1*n2, work, ldw, rdscal, dsum)
		pl = rdscal * math.Sqrt(dsum)
		if pl == 0 {
			pl = 1
		} else {
			pl = dscale / (math.Sqrt(dscale*dscale/pl+pl) * math.Sqrt(pl))
		}
		rdscal = 0
		dsum = 1
		rdscal, dsum = impl.Dlassq(n1*n2, work[n1*n2:], ldw, rdscal, dsum)
		pr = rdscal * math.Sqrt(dsum)
		if pr == 0 {
			pr = 1
		} else {
			pr = dscale / (math.Sqrt(dscale*dscale/pr+pr) * math.Sqrt(pr))
		}
	}

	if wantd {
		// compute estimates of Difu and Difl.

		if wantd1 {
			n1 := m
			n2 := n - m
			i := n1
			ijb := idifjob
			ldw := n1
			nw := n2
			// Frobenius norm-based Difu estimate.
			dif[0], dscale, ierr = impl.Dtgsyl(blas.NoTrans, ijb, n1, n2, a, lda, a[i*lda+i:], lda, work, ldw, b, ldb, b[i*ldb+i:], ldb, work[n1*n2:], ldw, work[2*n1*n2:], iwork, false)
			// Frobenius norm-based Difl estimate.
			dif[1], dscale, ierr = impl.Dtgsyl(blas.NoTrans, ijb, n2, n1, a[i*lda+i:], lda, a, lda, work, nw, b[i*ldb+i:], ldb, b, ldb, work[n1*n2:], nw, work[2*n1*n2:], iwork, false)

		} else {
			// Compute 1-norm-based estimates of Difu and Difl
			kase := 0
			n1 := m
			n2 := n - m
			i := n1
			ijb := 0
			mn2 := 2 * n1 * n2
			ldw := n1
			nw := n2
			// Difu norm estimate.

			for {
				dif[0], kase = impl.Dlacn2(mn2, work[mn2:], work, iwork, dif[0], kase, &isave)
				if kase == 0 {
					break
				}
				trans := blas.NoTrans
				if kase != 1 {
					trans = blas.Trans
				}
				dif[0], dscale, ierr = impl.Dtgsyl(trans, ijb, n1, n2, a, lda, a[i*lda+i:], lda, work, ldw, b, ldb, b[i*ldb+i:], ldb, work[n1*n2:], ldw, work[2*n1*n2:], iwork, false)
			}
			dif[0] = dscale / dif[0]

			// Difl norm estimate.

			for {
				dif[1], kase = impl.Dlacn2(mn2, work[mn2:], work, iwork, dif[1], kase, &isave)
				if kase == 0 {
					break
				}
				trans := blas.NoTrans
				if kase != 1 {
					trans = blas.Trans
				}
				dif[1], dscale, ierr = impl.Dtgsyl(trans, ijb, n2, n1, a[i*lda+i:], lda, a, lda, work, nw, b[i*ldb+i:], ldb, b, ldb, work[n1*n2:], nw, work[2*n1*n2:], iwork, false)
			}
			dif[1] = dscale / dif[1]
		}
	}

Sixty: // Check illCondition and lworkTooSmall flags.

	// Compute generalized eigenvalues of reordered pair (A,B)
	// and normalize the generalized Schur form.
	pair = false
	for k := 0; k < n; k++ {
		if pair {
			pair = false
		} else {
			if k < n-1 && a[(k+1)*lda+k] != 0 {
				pair = true
			}
			if pair {
				// Compute the eigenvalue(s) at position K.

				// TODO These feel like they will cause
				// problems for not being in row major storage?
				work[0] = a[k*lda+k]
				work[1] = a[(k+1)*lda+k]
				work[2] = a[k*lda+k+1]
				work[3] = a[(k+1)*lda+k+1]

				work[4] = b[k*ldb+k]
				work[5] = b[(k+1)*ldb+k]
				work[6] = b[k*ldb+k+1]
				work[7] = b[(k+1)*ldb+k+1]

				beta[k], beta[k+1], alphar[k], alphar[k+1], alphai[k] = impl.Dlag2(work, 2, work[4:], 2)
				alphai[k+1] = -alphai[k]
			} else {
				if math.Signbit(b[k*ldb+k]) {
					// Force b(k,k) non-negative.
					for i := 0; i < n; i++ {
						a[k*lda+i] *= -1
						b[k*ldb+i] *= -1
						if wantq {
							q[i*ldq+k] *= -1
						}
					}
				}
				alphar[k] = a[k*lda+k]
				alphai[k] = 0
				beta[k] = b[k*ldb+k]
			}
		}
	}
	work[0] = float64(lwmin)
	iwork[0] = liwmin
	return pl, pr, dif, m, illConditioned, lworkTooSmall, liworkTooSmall
}
