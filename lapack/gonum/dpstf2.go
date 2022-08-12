// Copyright ©2021 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dpstf2 computes the Cholesky factorization with complete pivoting of an n×n
// symmetric positive semidefinite matrix A.
//
// The factorization has the form
//  Pᵀ * A * P = Uᵀ * U ,  if uplo = blas.Upper,
//  Pᵀ * A * P = L  * Lᵀ,  if uplo = blas.Lower,
// where U is an upper triangular matrix and L is lower triangular, and P is
// stored as vector piv.
//
// Dpstf2 does not attempt to check that A is positive semidefinite.
//
// The length of piv must be n and the length of work must be at least 2*n,
// otherwise Dpstf2 will panic.
//
// Dpstf2 is an internal routine. It is exported for testing purposes.
func (Implementation) Dpstf2(uplo blas.Uplo, n int, a []float64, lda int, piv []int, tol float64, work []float64) (rank int, ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0, true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(piv) != n:
		panic(badLenPiv)
	case len(work) < 2*n:
		panic(shortWork)
	}

	// Initialize piv.
	for i := range piv[:n] {
		piv[i] = i
	}

	// Compute the first pivot.
	pvt := 0
	ajj := a[0]
	for i := 1; i < n; i++ {
		aii := a[i*lda+i]
		if aii > ajj {
			pvt = i
			ajj = aii
		}
	}
	if ajj <= 0 || math.IsNaN(ajj) {
		return 0, false
	}

	// Compute stopping value if not supplied.
	dstop := tol
	if dstop < 0 {
		dstop = float64(n) * dlamchE * ajj
	}

	// Set first half of work to zero, holds dot products.
	dots := work[:n]
	for i := range dots {
		dots[i] = 0
	}
	work2 := work[n : 2*n]

	bi := blas64.Implementation()
	if uplo == blas.Upper {
		// Compute the Cholesky factorization  Pᵀ * A * P = Uᵀ * U.
		for j := 0; j < n; j++ {
			// Update dot products and compute possible pivots which are stored
			// in the second half of work.
			for i := j; i < n; i++ {
				if j > 0 {
					tmp := a[(j-1)*lda+i]
					dots[i] += tmp * tmp
				}
				work2[i] = a[i*lda+i] - dots[i]
			}
			if j > 0 {
				// Find the pivot.
				pvt = j
				ajj = work2[pvt]
				for k := j + 1; k < n; k++ {
					wk := work2[k]
					if wk > ajj {
						pvt = k
						ajj = wk
					}
				}
				// Test for exit.
				if ajj <= dstop || math.IsNaN(ajj) {
					a[j*lda+j] = ajj
					return j, false
				}
			}
			if j != pvt {
				// Swap pivot rows and columns.
				a[pvt*lda+pvt] = a[j*lda+j]
				bi.Dswap(j, a[j:], lda, a[pvt:], lda)
				if pvt < n-1 {
					bi.Dswap(n-pvt-1, a[j*lda+(pvt+1):], 1, a[pvt*lda+(pvt+1):], 1)
				}
				bi.Dswap(pvt-j-1, a[j*lda+(j+1):], 1, a[(j+1)*lda+pvt:], lda)
				// Swap dot products and piv.
				dots[j], dots[pvt] = dots[pvt], dots[j]
				piv[j], piv[pvt] = piv[pvt], piv[j]
			}
			ajj = math.Sqrt(ajj)
			a[j*lda+j] = ajj
			// Compute elements j+1:n of row j.
			if j < n-1 {
				bi.Dgemv(blas.Trans, j, n-j-1,
					-1, a[j+1:], lda, a[j:], lda,
					1, a[j*lda+j+1:], 1)
				bi.Dscal(n-j-1, 1/ajj, a[j*lda+j+1:], 1)
			}
		}
	} else {
		// Compute the Cholesky factorization  Pᵀ * A * P = L * Lᵀ.
		for j := 0; j < n; j++ {
			// Update dot products and compute possible pivots which are stored
			// in the second half of work.
			for i := j; i < n; i++ {
				if j > 0 {
					tmp := a[i*lda+(j-1)]
					dots[i] += tmp * tmp
				}
				work2[i] = a[i*lda+i] - dots[i]
			}
			if j > 0 {
				// Find the pivot.
				pvt = j
				ajj = work2[pvt]
				for k := j + 1; k < n; k++ {
					wk := work2[k]
					if wk > ajj {
						pvt = k
						ajj = wk
					}
				}
				// Test for exit.
				if ajj <= dstop || math.IsNaN(ajj) {
					a[j*lda+j] = ajj
					return j, false
				}
			}
			if j != pvt {
				// Swap pivot rows and columns.
				a[pvt*lda+pvt] = a[j*lda+j]
				bi.Dswap(j, a[j*lda:], 1, a[pvt*lda:], 1)
				if pvt < n-1 {
					bi.Dswap(n-pvt-1, a[(pvt+1)*lda+j:], lda, a[(pvt+1)*lda+pvt:], lda)
				}
				bi.Dswap(pvt-j-1, a[(j+1)*lda+j:], lda, a[pvt*lda+(j+1):], 1)
				// Swap dot products and piv.
				dots[j], dots[pvt] = dots[pvt], dots[j]
				piv[j], piv[pvt] = piv[pvt], piv[j]
			}
			ajj = math.Sqrt(ajj)
			a[j*lda+j] = ajj
			// Compute elements j+1:n of column j.
			if j < n-1 {
				bi.Dgemv(blas.NoTrans, n-j-1, j,
					-1, a[(j+1)*lda:], lda, a[j*lda:], 1,
					1, a[(j+1)*lda+j:], lda)
				bi.Dscal(n-j-1, 1/ajj, a[(j+1)*lda+j:], lda)
			}
		}
	}
	return n, true
}
