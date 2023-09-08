// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

func (impl Implementation) Dgges(jobvsl, jobvsr lapack.UpdateSchurComp, sort lapack.Sort, selctg lapack.SelectGFunc, sense lapack.Sense, n int, a []float64, lda int, b []float64, ldb int, alphar, alphai, beta, vsl []float64, ldvsl int, vsr []float64, ldvsr int, work []float64, iwork []int, bwork []bool, isWorkspaceQuery bool) (sdim int, rconde, rcondv [2]float64, info int) {
	switch {
	case jobvsl != lapack.UpdateSchur && jobvsl != lapack.UpdateSchurNone:
		panic(badUpdateSchurComp)
	case jobvsr != lapack.UpdateSchur && jobvsr != lapack.UpdateSchurNone:
		panic(badUpdateSchurComp)
	case sort != lapack.SortExternal && sort != lapack.SortNone:
		panic(badSort)
	case sort == lapack.SortExternal && selctg == nil:
		panic(badSelectG)
	case sense != lapack.SenseNone && sense != lapack.SenseAverage && sense != lapack.SenseDeflated && sense != lapack.SenseBoth:
		panic(badSense)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldvsl < 1:
		panic(badLdVL)
	case ldvsr < 1:
		panic(badLdVR)
	case len(work) < 1:
		panic(shortWork)
	case len(iwork) < 1:
		panic(shortIWork)
	}

	if n == 0 {
		// Quick return for workspace query and normal return.
		work[0] = 1
		iwork[0] = 1
		return 0, rconde, rcondv, 0
	}

	var ijob, lwork, liwork int
	switch sense {
	case lapack.SenseNone:
		ijob = 0
	case lapack.SenseAverage:
		ijob = 1
	case lapack.SenseDeflated:
		ijob = 2
	case lapack.SenseBoth:
		ijob = 4
	}
	const spaceStr = " "
	minwrk := max(8*n, 6*n+16)
	maxwrk := minwrk - n + n*impl.Ilaenv(1, "DGEQRF", spaceStr, n, 1, n, 0)
	maxwrk = max(maxwrk, minwrk-n+n*impl.Ilaenv(1, "DORMQR", spaceStr, n, 1, n, -1))
	if jobvsl == lapack.UpdateSchur {
		maxwrk = max(maxwrk, minwrk-n+n*impl.Ilaenv(1, "DORGQR", spaceStr, n, 1, n, -1))
	}
	lwork = maxwrk
	if ijob >= 1 {
		lwork = max(lwork, n*n/2)
	}

	if isWorkspaceQuery {
		iwork[0] = 1
		if sense != lapack.SenseNone {
			iwork[0] = n + 6
		}
		work[0] = float64(lwork)
		return 0, rconde, rcondv, 0
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case len(alphar) < n:
		panic(badLenAlpha)
	case len(alphai) < n:
		panic(badLenAlpha)
	case len(beta) < n:
		panic(badLenBeta)
	case jobvsl == lapack.UpdateSchur && len(vsl) < (n-1)*ldvsl+n:
		panic(shortVL)
	case jobvsr == lapack.UpdateSchur && len(vsr) < (n-1)*ldvsr+n:
		panic(shortVR)
	case len(work) < lwork:
		panic(shortWork)
	case len(iwork) < liwork:
		panic(shortIWork)
	case len(bwork) < n:
		panic(shortBWork)
	}

	const (
		eps    = dlamchP
		safmin = dlamchS
		safmax = 1. / safmin
	)
	smlnum := math.Sqrt(safmin) / eps
	bignum := 1 / smlnum
	var (
		anrmto, bnrmto      float64
		ileft, iright, iwrk int
	)
	// Scale A if max element outside range smlnum..bignum.
	anrm := impl.Dlange(lapack.MaxAbs, n, n, a, lda, work)
	ilascl := false
	if anrm > 0 && anrm < smlnum {
		anrmto = smlnum
		ilascl = true
	} else {
		anrmto = bignum
		ilascl = true
	}
	if ilascl {
		impl.Dlascl(lapack.General, 0, 0, anrm, anrmto, n, n, a, lda)
	}

	bnrm := impl.Dlange(lapack.MaxAbs, n, n, b, ldb, work)
	ilbscl := false
	if bnrm > 0 && bnrm < smlnum {
		bnrmto = smlnum
		ilbscl = true
	} else {
		bnrmto = bignum
		ilbscl = true
	}
	if ilbscl {
		impl.Dlascl(lapack.General, 0, 0, bnrm, bnrmto, n, n, b, ldb)
	}

	// Permute the matrix to make it more nearly triangular.
	ileft = 0
	iright = n
	iwrk = iright + n
	// Dggbal
	// impl.Dggbal(jobvsl, jobvsr, n, a, lda, b, ldb, &ileft, &iright, work[iwrk:], iinfo)
	var ilo, ihi int
	// Reduce B to triangular form (QR decomposition of B).
	irows := ihi + 1 - ilo
	icols := n - ilo
	itau := iwrk
	iwrk = itau + irows
	impl.Dgeqrf(irows+1, icols+1, b[ilo*ldb+ilo:], ldb, work[itau:], work[iwrk:], lwork-iwrk)

	// Apply the orthogonal transformation to matrix A.
	impl.Dormqr(blas.Left, blas.Trans, irows+1, icols+1, irows+1, b[ilo*ldb+ilo:], ldb,
		work[itau:], a[ilo*lda+ilo:], lda, work[iwrk:], lwork-iwrk)

	// Initialize VSL.
	if jobvsl == lapack.UpdateSchur {
		impl.Dlaset(blas.All, n, n, 0, 1, vsl, ldvsl)
		if irows > 0 {
			impl.Dlacpy(blas.Lower, irows, irows, b[(ilo+1)*ldb+ilo:], ldb, vsl[(ilo+1)*ldvsl+ilo:], ldvsl)
		}
		impl.Dorgqr(irows+1, irows+1, irows+1, vsl[ilo*ldvsl+ilo:], ldvsl, work[itau:], work[iwrk:], lwork-iwrk)
	}

	// Initialize VSR.
	if jobvsr == lapack.UpdateSchur {
		impl.Dlaset(blas.All, n, n, 0, 1, vsr, ldvsr)
	}

	// Reduce to generalized Hessenberg form.
	// Dgghrd

	sdim = 0

	// Perform QZ algorithm, computing Schur vectors if desired.
	iwrk = itau
	// Dhgeqz
	var ierr int
	if ierr != 0 {
		// Handle error.
	}

	// Sort eigenvalues alpha/beta and compute the reciprocal of condition number(s).
	if sort != lapack.SortNone {
		// Undo scaling on eigenvalues before SELCTGing.
		if ilascl {
			impl.Dlascl(lapack.General, 0, 0, anrmto, anrm, n, 1, alphar, n)
			impl.Dlascl(lapack.General, 0, 0, anrmto, anrm, n, 1, alphai, n)
		}
		if ilbscl {
			impl.Dlascl(lapack.General, 0, 0, bnrmto, bnrm, n, 1, beta, n)
		}

		// Select Eigenvalues
		for i := 0; i < n; i++ {
			bwork[i] = selctg(complex(alphar[i], alphai[i]), beta[i])
		}

		// Reorder eigenvalues, transform Generalized Schur vectors,
		// and compute reciprocal condition numbers.
		var pl, pr float64
		var dif [2]float64
		// Dtgsen.
		if ijob >= 1 {
			maxwrk = max(maxwrk, 2*sdim*(n-sdim))
		}
		if ierr == -22 {
			// Handle error.
		} else {
			if ijob == 1 || ijob == 4 {
				rconde[0] = pl
				rconde[1] = pr
			}
			if ijob == 2 || ijob == 4 {
				rcondv[0] = dif[0]
				rcondv[1] = dif[1]
			}
			if ierr == 1 {
				info = n + 3
			}
		}
	}

	if jobvsl == lapack.UpdateSchur {
		// Apply back-permutation to VSL.
		// Dggbak.
	}
	if jobvsr == lapack.UpdateSchur {
		// Apply back-permutation to VSR.
		// Dggbak.
	}

	if ilascl {
		for i := 0; i < n; i++ {
			if alphai[i] == 0 {
				continue
			}
			bigR := alphar[i]/safmax > (anrmto/anrm) || safmin/alphar[i] > (anrm/anrmto)
			bigI := alphai[i]/safmax > (anrmto/anrm) || safmin/alphai[i] > (anrm/anrmto)
			if bigR {
				work[0] = math.Abs(a[i*lda+i] / alphar[i])
				beta[i] *= work[0]
				alphar[i] *= work[0]
				alphai[i] *= work[0]
			} else if bigI {
				work[0] = math.Abs(a[i*lda+i] / alphai[i])
				beta[i] *= work[0]
				alphar[i] *= work[0]
				alphai[i] *= work[0]
			}
		}
	}

	if ilbscl {
		for i := 0; i < n; i++ {
			if alphai[i] == 0 {
				continue
			}
			bigB := beta[i]/safmax > (bnrmto/bnrm) || safmin/beta[i] > (bnrm/bnrmto)
			if bigB {
				work[0] = math.Abs(b[i*ldb+i] / beta[i])
				beta[i] *= work[0]
				alphar[i] *= work[0]
				alphai[i] *= work[0]
			}
		}
	}

	// Undo scaling.
	if ilascl {
		impl.Dlascl(lapack.Hessenberg, 0, 0, anrmto, anrm, n, n, a, lda)
		impl.Dlascl(lapack.General, 0, 0, anrmto, anrm, n, 1, alphar, n)
		impl.Dlascl(lapack.General, 0, 0, anrmto, anrm, n, 1, alphai, n)
	}
	if ilbscl {
		impl.Dlascl(lapack.UpperTri, 0, 0, bnrmto, bnrm, n, n, b, ldb)
		impl.Dlascl(lapack.General, 0, 0, bnrmto, bnrm, n, 1, beta, n)
	}

	if sense == lapack.SenseNone {
		// Check if rendering is correct.

		lastsl := true
		lst2sl := true
		ip := 0
		sdim = 0
		var cursl bool
		for i := 0; i < n; i++ {
			cursl := selctg(complex(alphar[i], alphai[i]), beta[i])
			if alphai[i] == 0 {
				if cursl {
					sdim++
				}
				ip = 0
				if cursl && !lastsl {
					info = n + 2
				}
			} else {
				if ip == 0 {
					// Last eigenvalue conjugate pair.
					cursl = cursl || lastsl
					lastsl = cursl
					if cursl {
						sdim += 2
					}
					ip = -1
					if cursl && !lst2sl {
						info = n + 2
					}
				} else {
					// First eigenvalue of conjugate pair.
					ip = 0
				}
			}
			lst2sl = lastsl
			lastsl = cursl
		}
	}
	work[0] = float64(maxwrk)
	iwork[0] = liwork
	return sdim, rconde, rcondv, info
}
