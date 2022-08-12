// Copyright ©2021 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dtgsy2 solves the generalized Sylvester equation:
//  A * R - L * B = scale * C                (1)
//  D * R - L * E = scale * F,
// where R and L are unknown m×n matrices which on return are
// written into C and F, respectively.
//
// (A, D), (B, E) and (C, F) are given matrix pairs of size m×m,
// n×n and m×n, respectively. (A, D) and (B, E) must be in generalized
// Schur canonical form, i.e. A, B are upper
// quasi triangular and D, E are upper triangular.
//
// 0 <= scale <= 1 is an output scaling factor chosen to avoid overflow.
//
// rdsum and rdscal represent the sum of squares of computed contributions to
// the Dif-estimate from earlier solved sub-systems. rdscal is the scaling
// factor used to prevent overflow in rdsum. Dtgsy2 returns this sum of squares
// updated with the contributions from the current sub-system as scalout and sumout.
//
// if info is non-negative then Z (see below) was perturbed to avoid underflow during a call to Dgetc2.
//
// In matrix notation solving equation (1) corresponds to solve
// Z*x = scale*b, where Z is defined as
//  Z = [ kron(I_{n}, A)  -kron(Bᵀ, I_{m}) ]             (2)
//      [ kron(I_{n}, D)  -kron(Eᵀ, I_{m}) ],
// I_{k} is the identity matrix of size k and Xᵀ is the transpose of X.
// kron(X, Y) is the Kronecker product between the matrices X and Y.
// In the process of solving (1), we solve a number of such systems
// where Dim(In), Dim(In) = 1 or 2.
// If trans = blas.Trans, solve the transposed system Zᵀ*y = scale*b for y,
// which is equivalent to solve for R and L in
//  Aᵀ * R  + Dᵀ * L   = scale * C           (3)
//  R  * Bᵀ + L  * Eᵀ  = scale * -F
// This case is used to compute an estimate of Dif[(A, D), (B, E)] =
// sigma_min(Z) using reverse communication with Dlacon.
// Dtgsy2 also (ijob >= 1) contributes to the computation in Dtgsyl
// of an upper bound on the separation between to matrix pairs. Then
// the input (A, D), (B, E) are sub-pencils of the matrix pair in
// Dtgsyl. See Dtgsyl for details.
//
// Dtgsy2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dgghrd(compq byte, compz byte, n, ilo, ihi int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int) (info int) {
	//SUBROUTINE dgghrd( COMPQ, COMPZ, N, ILO, IHI, A, LDA, B, LDB, Q,LDQ, Z, LDZ, INFO )

	//  -- LAPACK computational routine --
	// -- LAPACK is a software package provided by Univ. of Tennessee,    --
	// -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
	//
	var ilq, ilz bool
	var icompq, icompz, jcol, jrow int
	_ = jcol
	_ = jrow
	var c, s, temp float64
	bi := blas64.Implementation()
	if compq == 'N' {
		ilq = false
		icompq = 1
	} else if compq == 'V' {
		ilq = true
		icompq = 2
	} else if compq == 'I' {
		ilq = true
		icompq = 3
	} else {
		icompq = 0
	}

	//     Decode COMPZ

	if compz == 'N' {
		ilz = false
		icompz = 1
	} else if compz == 'V' {
		ilz = true
		icompz = 2
	} else if compz == 'I' {
		ilz = true
		icompz = 3
	} else {
		icompz = 0
	}

	//     Test the input parameters.

	info = 0
	if icompq <= 0 {
		info = -1
	} else if icompz <= 0 {
		info = -2
	} else if n < 0 {
		info = -3
	} else if ilo < 1 {
		info = -4
	} else if ihi > n || ihi < ilo-1 {
		info = -5
	} else if lda < max(1, n) {
		info = -7
	} else if ldb < max(1, n) {
		info = -9
	} else if (ilq && (ldq < n)) || (ldq < 1) {
		info = -11
	} else if (ilz && ldz < n) || ldz < 1 {
		info = -13
	}
	if info != 0 {
		panic("DGGHRD")

	}

	//     Initialize Q and Z if desired.

	if icompq == 3 {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq) //CALL dlaset( 'Full', n, n, zero, one, q, ldq )
	}
	if icompz == 3 {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz) //CALL dlaset( 'Full', n, n, zero, one, z, ldz )
	}
	//     Quick return if possible

	if n < 1 {
		return
	}
	//     Zero out lower triangle of B

	for jcol := 1; jcol <= (n - 1); jcol++ { //DO 20 jcol = 1, n - 1
		for jrow := jcol + 1; jrow <= (n); jrow++ { //DO 10 jrow = jcol + 1, n
			b[(ldb*(jrow) - (ldb + 1) + jcol)] = 0 //b( jrow, jcol ) = zero
		} //10    CONTINUE
	} //  20 CONTINUE

	//     Reduce A and B

	for jcol := ilo; jcol <= ihi-2; jcol++ { //DO 40 jcol = ilo, ihi - 2

		for jrow := ihi; jrow >= jcol+2; jrow-- { //DO 30 jrow = ihi, jcol + 2, -1

			//           Step 1: rotate rows JROW-1, JROW to kill A(JROW,JCOL)

			temp = a[(lda*(jrow-1) - (lda + 1) + jcol)]                                                  //a( jrow-1, jcol )
			c, s, a[(lda*(jrow-1) - (lda + 1) + jcol)] = impl.Dlartg(temp, a[(lda*(jrow)-(lda+1)+jcol)]) //CALL dlartg( temp, a( jrow, jcol ), c, s,a( jrow-1, jcol ) )
			//func (impl Implementation) Dlartg(f, g float64) (cs, sn, r float64) {

			a[(lda*(jrow) - (lda + 1) + jcol)] = 0 //a( jrow, jcol ) = zero
			// trasponer matrices
			bi.Drot(n-jcol, a[(lda*(jrow-1)-(lda+1)+jcol+1):], lda, a[(lda*(jrow)-(lda+1)+jcol+1):], lda, c, s)
			//bi.Drot(n-jcol-1, a, lda, a, lda, c, s)
			//CALL drot( n-jcol,                   a( jrow-1, jcol+1 ), lda,                    a( jrow, jcol+1 ), lda,  c, s )
			bi.Drot(n+2+jrow, b[(ldb*(jrow-1)-(ldb+1)+jrow-1):], ldb, b[(ldb*(jrow)-(ldb+1)+jrow-1):], ldb, c, s) //CALL drot( n+2-jrow,                     b( jrow-1, jrow-1 ), ldb,                    b( jrow, jrow-1 ), ldb, c, s )
			if ilq {
				bi.Drot(n, q[(ldq*(1)-(ldq+1)+jrow-1):], 1, q[(ldq*(1)-(ldq+1)+jrow+1):], 1, c, s) //CALL drot( n,                     q( 1, jrow-1 ), 1,                      q( 1, jrow ), 1, c, s )
			}

			//           Step 2: rotate columns JROW, JROW-1 to kill B(JROW,JROW-1)

			temp = b[(ldb*(jrow) - (ldb + 1) + jrow)]                                                    //temp = b( jrow, jrow )
			c, s, b[(ldb*(jrow) - (ldb + 1) + jrow)] = impl.Dlartg(temp, b[(ldb*(jrow)-(ldb+1)+jrow+1)]) //CALL dlartg( temp, b( jrow, jrow-1 ), c, s,b( jrow, jrow ) )
			b[(ldb*(jrow) - (ldb + 1) + jrow - 1)] = 0                                                   //b( jrow, jrow-1 ) = zero
			bi.Drot(ihi, a[(lda*(1)-(lda+1)+jrow):], 1, a[(lda*(1)-(lda+1)+jrow-1):], 1, c, s)           //CALL drot( ihi,                     a( 1, jrow ), 1, a( 1, jrow-1 ), 1, c, s )
			bi.Drot(jrow-1, b[(ldb*(1)-(ldb+1)+jrow):], 1, b[(ldb*(1)-(ldb+1)+jrow-1):], 1, c, s)        //CALL drot( jrow-1,                     b( 1, jrow ), 1,                     b( 1, jrow-1 ), 1, c,s )
			if ilz {
				bi.Drot(n, z[(ldz*(1)-(ldz+1)+jrow):], 1, z[(ldz*(1)-(ldz+1)+jrow-1):], 1, c, s) //CALL drot( n,                     z( 1, jrow ), 1, z( 1, jrow-1 ), 1, c, s )
			}
		} //30    CONTINUE
	} //40 CONTINUE
	return info
}
