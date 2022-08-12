// Copyright ©2021 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
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
func (impl Implementation) Dtgsy2(trans blas.Transpose, ijob lapack.MaximizeNormXJob, m, n int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, d []float64, ldd int, e []float64, lde int, f []float64, ldf int, rdsum, rdscal float64, iwork []int, ipiv []int, jpiv []int) (scale, scalout, sumout float64, pq, info int) {
	//var IJOB, INFO, LDA, LDB, LDC, LDD, LDE, LDF, M, N,pq int
	//var   RDSCAL, RDSUM, SCALE float64
	//DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), C( LDC, * ),d( ldd, * ), e( lde, * ), f( ldf, * )
	//var LDZ int
	//LDZ = 8
	//var I, IE, IERR, II, IS, ISP1, J, JE, JJ, JS, JSP1,k, mb, nb, p, q, zdim int
	//var   ALPHA, SCALOC float64
	//var IPIV[LDZ] , JPIV[LDZ] []int
	//IPIV := make([]int, LDZ, LDZ)
	//jPIV := make([]int, LDZ, LDZ)
	//DOUBLE PRECISION   RHS( LDZ ), Z( LDZ, LDZ )
	//RHS:=  make([]float64, LDZ, LDZ)
	//ZHS:=  make([]float64, LDZ, LDZ)
	var p, i, is, q, j, scaloc, js, jsp1, je, nb, zdim, ie, isp1, mb, k, ii int
	var iwork2 []int
	iwork2 = make([]int, len(iwork))
	copy(iwork2, iwork)
	var alpha float64
	z := make([]float64, 64, 64)
	//ipiv := make([]int, 8, 8)
	//jpiv := make([]int, 8, 8)
	const ldz = 8
	var (
		rhss [ldz]float64
	)
	bi := blas64.Implementation()

	rhs := rhss[:]
	_ = scaloc
	var notran bool
	info = 0
	var ierr int
	ierr = 0
	notran = (trans == blas.NoTrans)
	if !notran && !(trans == blas.Trans) {
		info = -1
	} else if notran {
		if (ijob < 0) || (ijob > 2) {
			info = -2
		}
	}
	if info == 0 {
		if m <= 0 {
			info = -3
		} else if n <= 0 {
			info = -4
		} else if lda < max(1, m) {
			info = -6
		} else if ldb < max(1, n) {
			info = -8
		} else if ldc < max(1, m) {
			info = -10
		} else if ldd < max(1, m) {
			info = -12
		} else if lde < max(1, n) {
			info = -14
		} else if ldf < max(1, m) {
			info = -16
		}
	}
	if info != 0 {
		panic("DTGSY2, -info ")

	}
	pq = 0

	p = 0
	i = 1
G10: //CONTINUE
	if i > m {
		goto G20
	}
	p = p + 1
	iwork2[p-1] = i
	if i == m {
		goto G20
	}
	//if( a( i+1, i )!=0 ) THEN

	if a[(i+lda*(i-1))] != 0 {
		i = i + 2
	} else {
		i = i + 1
	}
	goto G10
G20:
	iwork2[p] = m + 1 //iwork( p+1 ) = m + 1

	q = p + 1 //q = p + 1
	j = 1
G30: //CONTINUE
	if j > n {
		goto G40
	}
	q = q + 1
	iwork2[q-1] = j
	if j == n {
		goto G40
	}
	//if( b( j+1, j )!=zero ) THEN
	if b[j+ldb*(j-1)] != 0 {
		j = j + 2
	} else {
		j = j + 1
	}
	goto G30
G40: //CONTINUE
	iwork2[q] = n + 1 //iwork( q+1 ) = n + 1
	pq = p * (q - p - 1)
	if notran {
		scale = 1
		scaloc = 1
		for j := (p + 2); j <= q; j++ { //DO 120 j = p + 2, q
			js = iwork2[j-1] //js = iwork( j )
			jsp1 = js + 1
			je = iwork2[j] - 1 //je = iwork( j+1 ) - 1
			nb = je - js + 1
			for i := p; i >= 1; i-- { //DO 110 i = p, 1, -1
				is = iwork2[i-1] //is = iwork( i )
				isp1 = is + 1
				ie = iwork2[i] - 1 //ie = iwork( i+1 ) - 1
				mb = ie - is + 1
				zdim = mb * nb * 2
				if (mb == 1) && (nb == 1) {
					//z[0] = a[is-1+lda*(is-1)]  //z( 1, 1 ) = a( is, is ) T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = d[is-1+ldd*(is-1)]  //z( 2, 1 ) = d( is, is )T
					z[8] = d[is-1+ldd*(is-1)]
					//z[8] = -b[js-1+ldb*(js-1)] //z( 1, 2 ) = -b( js, js )T
					z[1] = -b[js-1+ldb*(js-1)]
					//z[9] = -e[js-1+lde*(js-1)] //z( 2, 2 ) = -e( js, js )T
					z[9] = -e[js-1+lde*(js-1)]
					rhs[0] = c[is-1+ldc*(js-1)]
					rhs[1] = f[is-1+ldf*(js-1)]

					//zaux := make([]float64, 64)
					//var ix, jx int
					//for ix = 1; ix < 8; ix++ {
					//	for jx = 1; jx < 8; jx++ {
					//zaux[jx][ix] = z[ix][jx]
					//		zaux[jx-1+8*(ix-1)] = z[ix-1+8*(jx-1)]
					//	}
					//}

					k = impl.Dgetc2(zdim, z, ldz, ipiv[:(zdim)], jpiv[:(zdim)]) //CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					//k = impl.Dgetc2(zdim, z, ldz, ipiv, jpiv)

					if k > 0 {
						info = k
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv,scaloc )
						if scaloc != 1 {
							for k = 0; k < n; k++ { //DO 50 k = 1, n
								bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
								bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf) //CALL dscal( m, scaloc, f( 1, k ), 1 )
							} //G50:
							scale = scale * scaloc
						}
					} else {
						rdsum, rdscal = impl.Dlatdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv) //impl.Dlatdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv[:zdim], jpiv[:zdim])//CALL dlatdf( ijob, zdim, z, ldz, rhs, rdsum,rdscal, ipiv, jpiv )
					}
					c[is-1+ldc*(js-1)] = rhs[0] //c( is, js ) = rhs( 1 )
					f[is-1+ldf*(js-1)] = rhs[1] //f( is, js ) = rhs( 2 )
					if i > 1 {
						alpha = -rhs[0]                                                 //alpha = -rhs( 1 )
						bi.Daxpy(is+1, alpha, a[lda*(is-1):], lda, c[ldc*(js-1):], ldc) //CALL daxpy( is-1, alpha, a( 1, is ), 1, c( 1, js ), 1 )
						bi.Daxpy(is+1, alpha, d[ldd*(is-1):], ldd, f[ldf*(js-1):], ldf) //CALL daxpy( is-1, alpha, d( 1, is ), 1, f( 1, js ),1 )
					}
					if j < q {
						//x := []float64{-0.9999999999999998, -7, 7, 2, 2, 7, 1, -2, -2, -6, 12, -2, 8, 19, 0, 1, 0}
						//x := []float64{-0.9999999999999998, 7, -6, 19, 7, 1, 12, 0, 2, -2, -2, 1, 2, -2, 8, 0}
						//x := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
						bi.Daxpy(n-je, rhs[1], b[js-1+ldb*(je):], ldb, c[is-1+ldc*(je):], ldc) //CALL daxpy( n-je, rhs( 2 ), b( js, je+1 ), ldb,c( is, je+1 ), ldc )
						bi.Daxpy(n-je, rhs[1], e[js-1+lde*(je):], lde, f[is-1+ldf*(je):], ldf) //CALL daxpy( n-je, rhs( 2 ), e( js, je+1 ), lde,f( is, je+1 ), ldf )
					}

				} else if (mb == 1) && (nb == 2) {

					//z[0] = a[is-1+lda*(is-1)] //z( 1, 1 ) = a( is, is ) T aca
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = 0                  //z( 2, 1 ) = zeroT
					z[8] = 0
					//z[2] = d[is-1+ldd*(is-1)] //z( 3, 1 ) = d( is, is )T
					z[16] = d[is-1+ldd*(is-1)]
					//z[3] = 0                  //z( 4, 1 ) = zeroT
					z[24] = 0
					//z[8] = 0                   //z( 1, 2 ) = zeroT
					z[1] = 0
					//z[9] = a[is-1+lda*(is-1)]  //z( 2, 2 ) = a( is, is )T
					z[9] = a[is-1+lda*(is-1)]
					//z[10] = 0                  //z( 3, 2 ) = zeroT
					z[17] = 0
					//z[11] = d[is-1+ldd*(is-1)] //z( 4, 2 ) = d( is, is )T
					z[25] = d[is-1+ldd*(is-1)]
					//z[16] = -b[js-1+ldb*(js-1)]   //z( 1, 3 ) = -b( js, js )T
					z[2] = -b[js-1+ldb*(js-1)]
					//z[17] = -b[js-1+ldb*(jsp1-1)] //z( 2, 3 ) = -b( js, jsp1 )T
					z[10] = -b[js-1+ldb*(jsp1-1)]
					//z[18] = -e[js-1+lde*(js-1)]   //z( 3, 3 ) = -e( js, js )T
					z[18] = -e[js-1+lde*(js-1)]
					//z[19] = -e[js-1+lde*(jsp1-1)] //z( 4, 3 ) = -e( js, jsp1 )T
					z[26] = -e[js-1+lde*(jsp1-1)]
					//z[24] = -b[jsp1-1+ldb*(js-1)] //z( 1, 4 ) = -b( jsp1, js ) T
					z[3] = -b[jsp1-1+ldb*(js-1)]
					//z[25] = -b[jsp1-1+ldb*(jsp1-1)]//z( 2, 4 ) = -b( jsp1, jsp1 )T
					z[11] = -b[jsp1-1+ldb*(jsp1-1)]
					//z[26] = 0                      //z( 3, 4 ) = zero T
					z[19] = 0
					//z[27] = -e[jsp1-1+lde*(jsp1-1)]//z( 4, 4 ) = -e( jsp1, jsp1 ) T
					z[27] = -e[jsp1-1+lde*(jsp1-1)]
					rhs[0] = c[is-1+ldc*(js-1)]                                //rhs( 1 ) = c( is, js )
					rhs[1] = c[is-1+ldc*(jsp1-1)]                              //rhs( 2 ) = c( is, jsp1 )
					rhs[2] = f[is-1+ldf*(js-1)]                                //rhs( 3 ) = f( is, js )
					rhs[3] = f[is-1+ldf*(jsp1-1)]                              //rhs( 4 ) = f( is, jsp1 )
					ierr = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if ierr > 0 {
						info = ierr
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim])//CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv,scaloc )
						if scaloc != 1 {
							for k = 0; k < n; k++ { //DO 60 k = 1, n
								bi.Dscal(m, scaloc, c[k:], 1) //CALL dscal( m, scaloc, c( 1, k ), 1 )
								bi.Dscal(m, scaloc, f[k:], 1) //CALL dscal( m, scaloc, f( 1, k ), 1 )
							} //G60:                   }
							scale = scale * scaloc
						}
					} else {
						rdsum, rdscal = impl.Dlatdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv) //rdsum, rdscal = impl.Dlatdf(ijob, zdim, z, ldz, rhs,rdsum, rdscal, ipiv[:zdim], jpiv[:zdim])//CALL dlatdf( ijob, zdim, z, ldz, rhs, rdsum,rdscal, ipiv, jpiv )
					}
					c[is-1+ldc*(js-1)] = rhs[0]   //c( is, js ) = rhs( 1 )
					c[is-1+ldc*(jsp1-1)] = rhs[1] //c( is, jsp1 ) = rhs( 2 )
					f[is-1+ldf*(js-1)] = rhs[2]   //f( is, js ) = rhs( 3 )
					f[is-1+ldf*(jsp1-1)] = rhs[3] //f( is, jsp1 ) = rhs( 4 )
					if i > 1 {
						//bi.Dger(is-1, nb, -1, a[lda*(is-1):], 1, rhs[0:], 1, c[ldc*(js-1):], ldc)
						//CALL dger( is-1, nb, -one, a( 1, is ), 1, rhs( 1 ),1, c( 1, js ), ldc )
						impl.Dger2(is-1, nb, -1, a[lda*(is-1):], 1, rhs[0:], 1, c[ldc*(js-1):], ldc)
						//bi.Dger(is-1, nb, -1, d[ldd*(is-1):], 1, rhs[0:], 1, f[ldf*(js-1):], ldf)
						//CALL dger( is-1, nb, -one, d( 1, is ), 1, rhs( 1 ),1, f( 1, js ), ldf )
						impl.Dger2(is-1, nb, -1, d[ldd*(is-1):], 1, rhs[0:], 1, f[ldf*(js-1):], ldf)
					}
					if j < q {

						bi.Daxpy(n-je, rhs[2], b[js-1+ldb*(je):], ldb, c[is-1+ldc*(je):], ldc)   //CALL daxpy( n-je, rhs( 3 ), b( js, je+1 ), ldb,c( is, je+1 ), ldc )
						bi.Daxpy(n-je, rhs[2], e[js-1+lde*(je):], lde, f[is-1+ldf*(je):], ldf)   //CALL daxpy( n-je, rhs( 3 ), e( js, je+1 ), lde, f( is, je+1 ), ldf )
						bi.Daxpy(n-je, rhs[3], b[jsp1-1+ldb*(je):], ldb, c[is-1+ldc*(je):], ldc) //CALL daxpy( n-je, rhs( 4 ), b( jsp1, je+1 ), ldb, c( is, je+1 ), ldc )
						bi.Daxpy(n-je, rhs[3], e[jsp1-1+lde*(je):], lde, f[is-1+ldf*(je):], ldf) //CALL daxpy( n-je, rhs( 4 ), e( jsp1, je+1 ), lde,     f( is, je+1 ), ldf )
					}
				} else if (mb == 2) && (nb == 1) {
					//z[0] = a[is-1+lda*(is-1)]   //z( 1, 1 ) = a( is, is ) T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = a[isp1-1+lda*(is-1)] //z( 2, 1 ) = a( isp1, is )T
					z[8] = a[isp1-1+lda*(is-1)]
					//z[2] = d[is-1+ldd*(is-1)]   //z( 3, 1 ) = d( is, is )T
					z[16] = d[is-1+ldd*(is-1)]
					//z[3] = 0                    //z( 4, 1 ) = zero T
					z[24] = 0

					//z[8] = a[is-1+lda*(isp1-1)]   //z( 1, 2 ) = a( is, isp1 ) T
					z[1] = a[is-1+lda*(isp1-1)]
					//z[9] = a[isp1-1+lda*(isp1-1)] //z( 2, 2 ) = a( isp1, isp1 ) T
					z[9] = a[isp1-1+lda*(isp1-1)]
					//z[10] = d[is-1+ldd*(isp1-1)]  //z( 3, 2 ) = d( is, isp1 )  T
					z[17] = d[is-1+ldd*(isp1-1)]
					//z[11] = d[isp1-1+ldd*(isp1)]  //z( 4, 2 ) = d( isp1, isp1 ) T
					z[25] = d[isp1-1+ldd*(isp1)]

					//z[16] = -b[js-1+ldb*(js-1)] //z( 1, 3 ) = -b( js, js ) T
					z[2] = -b[js-1+ldb*(js-1)]
					//z[17] = 0                   //z( 2, 3 ) = zero T
					z[10] = 0
					//z[18] = -e[js-1+lde*(js-1)] //z( 3, 3 ) = -e( js, js ) T
					z[18] = -e[js-1+lde*(js-1)]
					//z[19] = 0                   //z( 4, 3 ) = zero T
					z[26] = 0
					//z[24] = 0                   //z( 1, 4 ) = zero T
					z[3] = 0
					//z[25] = -b[js-1+ldb*(js-1)] //z( 2, 4 ) = -b( js, js ) T
					z[11] = -b[js-1+ldb*(js-1)]
					//z[26] = 0                   //z( 3, 4 ) = zero T
					z[19] = 0
					//z[27] = -e[js-1+lde*(js-1)] //z( 4, 4 ) = -e( js, js ) T
					z[27] = -e[js-1+lde*(js-1)]
					rhs[0] = c[is-1+ldc*(js-1)]                             //rhs( 1 ) = c( is, js )
					rhs[1] = c[isp1-1+ldc*(js-1)]                           //rhs( 2 ) = c( isp1, js )
					rhs[2] = f[is-1+ldf*(js-1)]                             //rhs( 3 ) = f( is, js )
					rhs[3] = f[isp1-1+ldf*(js-1)]                           //rhs( 4 ) = f( isp1, js )
					k = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if k > 0 {
						info = ierr
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv, scaloc )
						if scaloc != 1 {
							for k = 0; k < n; k++ { //DO 70 k = 1, n
								bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
								bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf) //CALL dscal( m, scaloc, f( 1, k ), 1 )
							}
							scale = scale * scaloc
						}
					} else {
						rdsum, rdscal = impl.Dlatdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv) //CALL dlatdf( ijob, zdim, z, ldz, rhs, rdsum,rdscal, ipiv, jpiv )
					}

					c[is-1+ldc*(js-1)] = rhs[0]   //c( is, js ) = rhs( 1 )
					c[isp1-1+ldc*(js-1)] = rhs[1] //c( isp1, js ) = rhs( 2 )
					f[is-1+ldf*(js-1)] = rhs[2]   //f( is, js ) = rhs( 3 )
					f[isp1-1+ldf*(js-1)] = rhs[3] //f( isp1, js ) = rhs( 4 )
					if i > 1 {

						//bi.Dgemv(blas.NoTrans, is-1, mb, -1, a[lda*(is-1):], lda, rhs, 1, 1, c[ldc*(js-1):], 1)
						impl.Dgemv2('N', is-1, mb, -1, a[lda*(is-1):], lda, rhs[0:], 1, 1, c[ldc*(js-1):], 1)
						//CALL dgemv( 'N'     , is-1, mb, -one, a( 1, is ), lda,  rhs( 1 ), 1, one, c( 1, js ), 1 )
						//bi.Dgemv(blas.NoTrans, is-1, mb, -1, d[ldd*(is-1):], ldd, rhs, 1, 1, f[ldf*(js-1):], ldf) //CALL dgemv( 'N', is-1, mb, -one, d( 1, is ), ldd, rhs( 1 ), 1, one, f( 1, js ), 1 )
						impl.Dgemv2('N', is-1, mb, -1, d[ldd*(is-1):], ldd, rhs[0:], 1, 1, f[ldf*(js-1):], 1)
					}
					if j < q {
						baux := make([]float64, 16)
						var ix, jx int
						for ix = 1; ix <= 4; ix++ {
							for jx = 1; jx <= 4; jx++ {
								baux[jx-1+ldb*(ix-1)] = b[ix-1+ldb*(jx-1)]
							}
						}
						caux := make([]float64, 16)

						for ix = 1; ix <= 4; ix++ {
							for jx = 1; jx <= 4; jx++ {
								caux[jx-1+ldc*(ix-1)] = c[ix-1+ldc*(jx-1)]
							}
						}
						//bi.Dger(mb, n-je, 1, rhs[2:], 1, b[js-1+ldb*(je):], ldb, c[is-1+ldc*(je):], ldc)
						impl.Dger2(mb, n-je, 1, rhs[2:], 1, b[js-1+ldb*(je):], ldb, c[is-1+ldc*(je):], ldc)
						//CALL dger( mb, n-je, one, rhs( 3 ), 1,b( js, je+1 ), ldb, c( is, je+1 ), ldc )
						impl.Dger2(mb, n-je, 1, rhs[2:], 1, e[js-1+lde*(je):], lde, f[is-1+ldf*(je):], ldf)
						//CALL dger( mb, n-je, one, rhs( 3 ), 1,e( js, je+1 ), lde, f( is, je+1 ), ldf )
					}
				} else if (mb == 2) && (nb == 2) {
					impl.Dlaset(blas.All, ldz, ldz, 0, 0, z, ldz) //CALL dlaset( 'F', ldz, ldz, zero, zero, z, ldz )

					//z[0] = a[is-1+lda*(is-1)]   //z( 1, 1 ) = a( is, is )  T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = a[isp1-1+lda*(is-1)] //z( 2, 1 ) = a( isp1, is )T
					z[8] = a[isp1-1+lda*(is-1)]
					//z[4] = d[is-1+ldd*(is-1)]   //z( 5, 1 ) = d( is, is )T
					z[32] = d[is-1+ldd*(is-1)]

					//z[8] = a[is-1+lda*(isp1-1)]    //z( 1, 2 ) = a( is, isp1 )T
					z[1] = a[is-1+lda*(isp1-1)]
					//z[9] = a[isp1-1+lda*(isp1-1)]  //z( 2, 2 ) = a( isp1, isp1 )T
					z[9] = a[isp1-1+lda*(isp1-1)]
					//z[12] = d[is-1+ldd*(isp1-1)]   //z( 5, 2 ) = d( is, isp1 )T
					z[33] = d[is-1+ldd*(isp1-1)]
					//z[13] = d[isp1-1+ldd*(isp1-1)] //z( 6, 2 ) = d( isp1, isp1 )T
					z[41] = d[isp1-1+ldd*(isp1-1)]
					//z[16] = a[is-1+lda*(is-1)]   //z( 3, 3 ) = a( is, is )T
					z[18] = a[is-1+lda*(is-1)]
					//z[17] = a[isp1-1+lda*(is-1)] //z( 4, 3 ) = a( isp1, is )T
					z[26] = a[isp1-1+lda*(is-1)]
					//z[18] = d[is-1+ldd*(is-1)]   //z( 7, 3 ) = d( is, is )T
					z[50] = d[is-1+ldd*(is-1)]
					//z[26] = a[is-1+lda*(isp1-1)]   //z( 3, 4 ) = a( is, isp1 )T
					z[19] = a[is-1+lda*(isp1-1)]
					//z[27] = a[isp1-1+lda*(isp1-1)] //z( 4, 4 ) = a( isp1, isp1 )T
					z[27] = a[isp1-1+lda*(isp1-1)]
					//z[30] = d[is-1+ldd*(isp1-1)]   //z( 7, 4 ) = d( is, isp1 )T
					z[51] = d[is-1+ldd*(isp1-1)]
					//z[31] = d[isp1-1+ldd*(isp1-1)] //z( 8, 4 ) = d( isp1, isp1 )T
					z[59] = d[isp1-1+ldd*(isp1-1)]
					//z[32] = -b[js-1+ldb*(js-1)]   //z( 1, 5 ) = -b( js, js )T
					z[4] = -b[js-1+ldb*(js-1)]
					//z[34] = -b[js-1+ldb*(jsp1-1)] //z( 3, 5 ) = -b( js, jsp1 )T
					z[20] = -b[js-1+ldb*(jsp1-1)]
					//z[36] = -e[js-1+lde*(js-1)]   //z( 5, 5 ) = -e( js, js )T
					z[36] = -e[js-1+lde*(js-1)]
					//z[38] = -e[js-1+lde*(jsp1-1)] //z( 7, 5 ) = -e( js, jsp1 )T
					z[52] = -e[js-1+lde*(jsp1-1)]
					//z[41] = -b[js-1+ldb*(js-1)]   //z( 2, 6 ) = -b( js, js )T
					z[13] = -b[js-1+ldb*(js-1)]
					//z[43] = -b[js-1+ldb*(jsp1-1)] //z( 4, 6 ) = -b( js, jsp1 )T
					z[29] = -b[js-1+ldb*(jsp1-1)]
					//z[45] = -e[js-1+lde*(js-1)]   //z( 6, 6 ) = -e( js, js )T
					z[45] = -e[js-1+lde*(js-1)]
					//z[47] = -e[js-1+lde*(jsp1-1)] //z( 8, 6 ) = -e( js, jsp1 )T
					z[61] = -e[js-1+lde*(jsp1-1)]
					//z[48] = -b[jsp1-1+ldb*(js-1)]   //z( 1, 7 ) = -b( jsp1, js )T
					z[6] = -b[jsp1-1+ldb*(js-1)]
					//z[50] = -b[jsp1-1+ldb*(jsp1-1)] //z( 3, 7 ) = -b( jsp1, jsp1 )T
					z[22] = -b[jsp1-1+ldb*(jsp1-1)]
					//z[54] = -e[jsp1-1+lde*(jsp1-1)] //z( 7, 7 ) = -e( jsp1, jsp1 )T
					z[54] = -e[jsp1-1+lde*(jsp1-1)]
					//z[57] = -b[jsp1-1+ldb*(js-1)]   //z( 2, 8 ) = -b( jsp1, js )T
					z[15] = -b[jsp1-1+ldb*(js-1)]
					//z[59] = -b[jsp1-1+ldb*(jsp1-1)] //z( 4, 8 ) = -b( jsp1, jsp1 )T
					z[31] = -b[jsp1-1+ldb*(jsp1-1)]
					//z[63] = -e[jsp1-1+lde*(jsp1-1)] //z( 8, 8 ) = -e( jsp1, jsp1 )T
					z[63] = -e[jsp1-1+lde*(jsp1-1)]
					k = 1
					ii = mb*nb + 1
					for jj := 0; jj <= nb-1; jj++ { //DO 80 jj = 0, nb - 1
						bi.Dcopy(mb, c[is-1+ldc*(js+jj-1):], 1, rhs[k-1:], 1)  //CALL dcopy( mb, c( is, js+jj ), 1, rhs( k ), 1 )
						bi.Dcopy(mb, f[is-1+ldf*(js+jj-1):], 1, rhs[ii-1:], 1) //CALL dcopy( mb, f( is, js+jj ), 1, rhs( ii ), 1 )
						k = k + mb
						ii = ii + mb
						//CONTINUE
					}
					k = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if k > 0 {
						info = ierr
					}
					if ijob == 0 {
						scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv, scaloc )
						if scaloc != 1 {
							for k = 0; k < n; k++ { //DO 90 k = 1, n
								bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
								bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf) //CALL dscal( m, scaloc, f( 1, k ), 1 )
							} //CONTINUE//G90:
							scale = scale * scaloc
						}
					} else {
						rdsum, rdscal = impl.Dlatdf(ijob, zdim, z, ldz, rhs, rdsum, rdscal, ipiv, jpiv) //impl.Dlatdf(ijob, zdim, z, ldz, rhs,rdsum, rdscal, ipiv[:zdim], jpiv[:zdim])//CALL dlatdf( ijob, zdim, z, ldz, rhs, rdsum,       rdscal, ipiv, jpiv )
					}
					k = 0
					ii = mb * nb
					for jj := 0; jj <= nb-1; jj++ { //DO 100 jj = 0, nb - 1
						bi.Dcopy(mb, rhs[k:], 1, c[is-1+ldc*(js+jj-1):], 1)  //CALL dcopy( mb, rhs( k ), 1, c( is, js+jj ), 1 )
						bi.Dcopy(mb, rhs[ii:], 1, f[is-1+ldf*(js+jj-1):], 1) //CALL dcopy( mb, rhs( ii ), 1, f( is, js+jj ), 1 )
						k = k + mb
						ii = ii + mb
					} //G100:
					if i > 1 {
						//bi.Dgemm(blas.NoTrans, blas.NoTrans, is-1, nb, mb, -1, a[lda*(is-1):], lda, rhs, mb, 1, c[ldc*(js-1):], ldc) // CALL dgemm(    'N',          'N', is-1, nb, mb, -one, a( 1, is ), lda, rhs( 1 ), mb, one,  c( 1, js ), ldc )
						//bi.Dgemm(blas.NoTrans, blas.NoTrans, is-1, nb, mb, -1, d[ldd*(is-1):], ldd, rhs, mb, 1, f[ldf*(js-1):], ldf) //CALL dgemm(     'N',          'N', is-1, nb, mb, -one, d( 1, is ), ldd, rhs( 1 ), mb, one,  f( 1, js ), ldf )

						impl.Dgemm2('N', 'N', is-1, nb, mb, -1, a[lda*(is-1):], lda, rhs, mb, 1, c[ldc*(js-1):], ldc) // CALL dgemm(    'N',          'N', is-1, nb, mb, -one, a( 1, is ), lda, rhs( 1 ), mb, one,  c( 1, js ), ldc )
						impl.Dgemm2('N', 'N', is-1, nb, mb, -1, d[ldd*(is-1):], ldd, rhs, mb, 1, f[ldf*(js-1):], ldf) //CALL dgemm(     'N',          'N', is-1, nb, mb, -one, d( 1, is ), ldd, rhs( 1 ), mb, one,  f( 1, js ), ldf )
					}
					if j < q {
						k = mb * nb
						//bi.Dgemm(blas.NoTrans, blas.NoTrans, mb, n-je, nb, 1, rhs[k:], mb, b[js-1+ldb*(je):], ldb, 1, c[is-1+ldc*(je):], ldc) //CALL dgemm('N','N', mb, n-je, nb, one, rhs( k ), mb,             b( js, je+1 ), ldb, one,c( is, je+1 ), ldc )
						//bi.Dgemm(blas.NoTrans, blas.NoTrans, mb, n-je, nb, 1, rhs[k:], mb, e[js-1+lde*(je):], lde, 1, f[is-1+ldf*(je):], ldf) //CALL dgemm('N','N', mb, n-je, nb, one, rhs( k ), mb,e( js, je+1 ), lde, one, f( is, je+1 ), ldf )
						impl.Dgemm2('N', 'N', mb, n-je, nb, 1, rhs[k:], mb, b[js-1+ldb*(je):], ldb, 1, c[is-1+ldc*(je):], ldc) //CALL dgemm('N','N', mb, n-je, nb, one, rhs( k ), mb,             b( js, je+1 ), ldb, one,c( is, je+1 ), ldc )
						impl.Dgemm2('N', 'N', mb, n-je, nb, 1, rhs[k:], mb, e[js-1+lde*(je):], lde, 1, f[is-1+ldf*(je):], ldf) //CALL dgemm('N','N', mb, n-je, nb, one, rhs( k ), mb,e( js, je+1 ), lde, one, f( is, je+1 ), ldf )

					}

				}

			} //G110:
		} //G120:
	} else {
		scale = 1
		scaloc = 1
		for i := 0; i <= p; i++ { //DO 200 i = 1, p

			is = iwork2[i]
			isp1 = is + 1
			ie = iwork2[i+1] - 1
			mb = ie - is + 1
			for j := q; j >= p+2; j-- { //DO 190 j = q, p + 2, -1
				js = iwork2[j]
				jsp1 = js + 1
				je = iwork2[j+1] - 1
				nb = je - js + 1
				zdim = mb * nb * 2
				if (mb == 1) && (nb == 1) {
					//z[0] = a[is-1+lda*(is-1)]                 //z( 1, 1 ) = a( is, is ) T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = -b[js-1+ldb*(js-1)]                //z( 2, 1 ) = -b( js, js )T
					z[8] = -b[js-1+ldb*(js-1)]
					//z[8] = d[is-1+ldd*(is-1)]                 //z( 1, 2 ) = d( is, is )T
					z[1] = d[is-1+ldd*(is-1)]
					//z[9] = -e[js-1+lde*(js-1)]                //z( 2, 2 ) = -e( js, js )T
					z[9] = -e[js-1+lde*(js-1)]
					rhs[0] = c[is-1+ldc*(js-1)]                             //rhs( 1 ) = c( is, js )
					rhs[1] = f[is-1+ldf*(js-1)]                             //rhs( 2 ) = f( is, js )
					k = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim])//CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if k > 0 {
						info = ierr
					}
					scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim])//CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv, scaloc )
					if scaloc != 1 {
						for k = 0; k < n; k++ { //DO 130 k = 1, n
							bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
							bi.Dscal(m, scaloc, f[ldf*(k-1):], ldc) //CALL dscal( m, scaloc, f( 1, k ), 1 )
						} //CONTINUE //G130:
						scale = scale * scaloc
					}
					c[is-1+ldc*(js-1)] = rhs[0] //c( is, js ) = rhs( 1 )
					f[is-1+ldf*(js-1)] = rhs[1] //f( is, js ) = rhs( 2 )
					if j > p+2 {
						alpha = rhs[0]
						bi.Daxpy(js, alpha, b[ldb+(js-1):], ldb, f[is-1:], ldf) //CALL daxpy( js-1, alpha, b( 1, js ), 1, f( is, 1 ), ldf )  aca quede
						alpha = rhs[1]
						bi.Daxpy(js, alpha, e[lde*(js-1):], lde, f[is-1:], ldf) //CALL daxpy( js-1, alpha, e( 1, js ), 1, f( is, 1 ), ldf )
					}
					if i < p {
						alpha = -rhs[0]
						bi.Daxpy(m-ie+1, alpha, a[is-1+lda*(ie):], lda, c[ie+ldc*(js-1):], 1) //CALL daxpy( m-ie, alpha, a( is, ie+1 ), lda,   c( ie+1, js ), 1 )
						alpha = -rhs[1]
						bi.Daxpy(m-ie+1, alpha, d[is-1+ldd*(ie):], ldd, c[ie+ldc*(js-1):], 1) //CALL daxpy( m-ie, alpha, d( is, ie+1 ), ldd, c( ie+1, js ), 1 )
					}
				} else if (mb == 1) && (nb == 2) {
					//z[0] = a[is-1+lda*(is-1)]    //z( 1, 1 ) = a( is, is ) T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = 0                     //z( 2, 1 ) = zero T
					z[8] = 0
					//z[2] = -b[js-1+ldb*(js-1)]   //z( 3, 1 ) = -b( js, js ) T
					z[16] = -b[js-1+ldb*(js-1)]
					//z[3] = -b[jsp1-1+ldb*(js-1)] //z( 4, 1 ) = -b( jsp1, js ) T
					z[24] = -b[jsp1-1+ldb*(js-1)]
					//z[8] = 0                        //z( 1, 2 ) = zero T
					z[1] = 0
					//z[9] = a[is-1+lda*(is-1)]       //z( 2, 2 ) = a( is, is ) T
					z[9] = a[is-1+lda*(is-1)]
					//z[10] = -b[js-1+ldb*(jsp1-1)]   //z( 3, 2 ) = -b( js, jsp1 ) T
					z[17] = -b[js-1+ldb*(jsp1-1)]
					//z[11] = -b[jsp1-1+ldb*(jsp1-1)] //z( 4, 2 ) = -b( jsp1, jsp1 ) T
					z[25] = -b[jsp1-1+ldb*(jsp1-1)]
					//z[16] = d[is-1+ldd*(is-1)]  //z( 1, 3 ) = d( is, is ) T
					z[2] = d[is-1+ldd*(is-1)]
					//z[17] = 0                   //z( 2, 3 ) = zero T
					z[10] = 0
					//z[18] = -e[js-1+lde*(js-1)] //z( 3, 3 ) = -e( js, js ) T
					z[18] = -e[js-1+lde*(js-1)]
					//z[19] = 0                   //z( 4, 3 ) = zero T
					z[26] = 0
					//z[24] = 0                                 //z( 1, 4 ) = zero T
					z[3] = 0
					//z[25] = d[is-1+ldd*(is-1)]                //z( 2, 4 ) = d( is, is ) T
					z[11] = d[is-1+ldd*(is-1)]
					//z[26] = -e[js-1+lde*(jsp1-1)]             //z( 3, 4 ) = -e( js, jsp1 ) T
					z[19] = -e[js-1+lde*(jsp1-1)]
					//z[27] = -e[jsp1-1+lde*(jsp1-1)]           //z( 4, 4 ) = -e( jsp1, jsp1 ) T
					z[27] = -e[jsp1-1+lde*(jsp1-1)]
					rhs[0] = c[is-1+ldc*(js-1)]                             //rhs( 1 ) = c( is, js )
					rhs[1] = c[is-1+ldc*(jsp1-1)]                           //rhs( 2 ) = c( is, jsp1 )
					rhs[2] = f[is-1+ldf*(js-1)]                             //rhs( 3 ) = f( is, js )
					rhs[3] = f[is-1+ldf*(jsp1-1)]                           //rhs( 4 ) = f( is, jsp1 )
					k = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim])//CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if k > 0 {
						info = ierr
					}
					scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv, scaloc )
					if scaloc != 1 {
						for k = 0; k < n; k++ { //DO 140 k = 1, n
							bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
							bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf) //CALL dscal( m, scaloc, f( 1, k ), 1 )
						} //G140:
						scale = scale * scaloc
					}
					c[is-1+ldc*(js-1)] = rhs[0]   //c( is, js ) = rhs( 1 )
					c[is-1+ldc*(jsp1-1)] = rhs[1] //c( is, jsp1 ) = rhs( 2 )
					f[is-1+ldf*(js-1)] = rhs[2]   //f( is, js ) = rhs( 3 )
					f[is-1+ldf*(jsp1-1)] = rhs[3] //f( is, jsp1 ) = rhs( 4 )
					if j > p+2 {
						bi.Daxpy(js, rhs[0], b[ldb*(js-1):], ldb, f[is-1:], ldf)   //CALL daxpy( js-1, rhs( 1 ), b( 1, js ), 1, f( is, 1 ), ldf )
						bi.Daxpy(js, rhs[1], b[ldb*(jsp1-1):], ldb, f[is-1:], ldf) //CALL daxpy( js-1, rhs( 2 ), b( 1, jsp1 ), 1,f( is, 1 ), ldf )
						bi.Daxpy(js, rhs[2], e[lde*(js-1):], lde, f[is-1:], ldf)   //CALL daxpy( js-1, rhs( 3 ), e( 1, js ), 1,f( is, 1 ), ldf )
						bi.Daxpy(js, rhs[3], e[lde*(jsp1-1):], lde, f[is-1:], ldf) //CALL daxpy( js-1, rhs( 4 ), e( 1, jsp1 ), 1,f( is, 1 ), ldf )
					}
					if i < p {
						//bi.Dger(m-ie-1, nb, -1, a[is-1+lda*(ie):], lda, rhs, 1, c[ie+ldc*(js-1):], ldc) //CALL dger( m-ie, nb, -one, a( is, ie+1 ), lda,rhs( 1 ), 1, c( ie+1, js ), ldc )
						//bi.Dger(m-ie-1, nb, -1, d[is-1+ldd*(ie):], ldd, rhs, 1, c[ie+ldc*(js-1):], ldc) //CALL dger( m-ie, nb, -one, d( is, ie+1 ), ldd, rhs( 3 ), 1, c( ie+1, js ), ldc )
						impl.Dger2(m-ie-1, nb, -1, a[is-1+lda*(ie):], lda, rhs[0:], 1, c[ie+ldc*(js-1):], ldc) //CALL dger( m-ie, nb, -one, a( is, ie+1 ), lda,rhs( 1 ), 1, c( ie+1, js ), ldc )
						impl.Dger2(m-ie-1, nb, -1, d[is-1+ldd*(ie):], ldd, rhs[2:], 1, c[ie+ldc*(js-1):], ldc) //CALL dger( m-ie, nb, -one, d( is, ie+1 ), ldd, rhs( 3 ), 1, c( ie+1, js ), ldc )

					}
				} else if (mb == 2) && (nb == 1) {
					//z[0] = a[is-1+lda*(is-1)]   //z( 1, 1 ) = a( is, is ) T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = a[is-1+lda*(isp1-1)] //z( 2, 1 ) = a( is, isp1 ) T
					z[8] = a[is-1+lda*(isp1-1)]
					//z[2] = -b[js-1+ldb*(js-1)]  //z( 3, 1 ) = -b( js, js ) T
					z[16] = -b[js-1+ldb*(js-1)]
					//z[3] = 0                    //z( 4, 1 ) = zero
					z[24] = 0
					//z[8] = a[isp1-1+lda*(is-1)]   //z( 1, 2 ) = a( isp1, is ) T
					z[1] = a[isp1-1+lda*(is-1)]
					//z[9] = a[isp1-1+lda*(isp1-1)] //z( 2, 2 ) = a( isp1, isp1 ) T
					z[9] = a[isp1-1+lda*(isp1-1)]
					//z[10] = 0                     //z( 3, 2 ) = zero T
					z[17] = 0
					//z[11] = -b[js-1+ldb*(js-1)]   //z( 4, 2 ) = -b( js, js ) T
					z[25] = -b[js-1+ldb*(js-1)]
					//z[16] = d[is-1+ldd*(is-1)]   //z( 1, 3 ) = d( is, is ) T
					z[2] = d[is-1+ldd*(is-1)]
					//z[17] = d[is-1+ldd*(isp1-1)] //z( 2, 3 ) = d( is, isp1 ) T
					z[10] = d[is-1+ldd*(isp1-1)]
					//z[18] = -e[js-1+lde*(js-1)]  //z( 3, 3 ) = -e( js, js ) T
					z[18] = -e[js-1+lde*(js-1)]
					//z[19] = 0                    //z( 4, 3 ) = zero T
					z[26] = 0
					//z[24] = 0                                 //z( 1, 4 ) = zero T
					z[3] = 0
					//z[25] = d[isp1-1+ldd*(isp1-1)]            //z( 2, 4 ) = d( isp1, isp1 ) T
					z[11] = d[isp1-1+ldd*(isp1-1)]
					//z[26] = 0                                 //z( 3, 4 ) = zero T
					z[19] = 0
					//z[27] = -e[js-1+lde*(js-1)]               //z( 4, 4 ) = -e( js, js ) T
					z[27] = -e[js-1+lde*(js-1)]
					rhs[0] = c[is-1+ldc*(js-1)]                             //rhs( 1 ) = c( is, js )
					rhs[1] = c[isp1-1+ldc*(js-1)]                           //rhs( 2 ) = c( isp1, js )
					rhs[2] = f[is-1+ldf*(js-1)]                             //rhs( 3 ) = f( is, js )
					rhs[3] = f[isp1-1+ldf*(js-1)]                           //rhs( 4 ) = f( isp1, js )
					k = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if k > 0 {
						info = ierr
					}
					scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv, scaloc )
					if scaloc != 1 {
						for k = 0; k < n; k++ { //DO 150 k = 1, n
							bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
							bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf) //CALL dscal( m, scaloc, f( 1, k ), 1 )
						} //CONTINUE//G150:
						scale = scale * scaloc
					}
					c[is-1+ldc*(js-1)] = rhs[0]   //c( is, js ) = rhs( 1 )
					c[isp1-1+ldc*(js-1)] = rhs[1] //c( isp1, js ) = rhs( 2 )
					f[is-1+ldf*(js-1)] = rhs[2]   //f( is, js ) = rhs( 3 )
					f[isp1-1+ldf+(js-1)] = rhs[0] //f( isp1, js ) = rhs( 4 )
					if j > p+2 {
						//bi.Dger(mb, js, 1, rhs, 1, b[ldb*(js-1):], ldb, f[is-1:], ldf)
						impl.Dger2(mb, js-1, 1, rhs[0:], 1, b[ldb*(js-1):], ldb, f[is-1:], ldf)
						//CALL dger( mb, js-1, one, rhs( 1 ), 1, b( 1, js ),1, f( is, 1 ), ldf )
						impl.Dger2(mb, js, 1, rhs[2:], 1, e[lde*(js-1):], lde, f[is-1:], ldf)
						//CALL dger( mb, js-1, one, rhs( 3 ), 1, e( 1, js ), 1, f( is, 1 ), ldf )
					}
					if i < p {
						//bi.Dgemv(blas.Trans, mb, m-ie, -1, a[is-1+lda*(ie):], lda, rhs, 1, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemv( 'T', mb, m-ie, -one,a( is, ie+1 ),lda, rhs( 1 ), 1, one, c( ie+1, js ),1 )
						//bi.Dgemv(blas.Trans, mb, m-ie, -1, d[is-1+ldd*(ie):], ldd, rhs, 1, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemv( 'T', mb, m-ie, -one,d( is, ie+1 ),ldd , rhs( 3 ), 1, one, c( ie+1, js ),1 )
						impl.Dgemv2('T', mb, m-ie, -1, a[is-1+lda*(ie):], lda, rhs[0:], 1, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemv( 'T', mb, m-ie, -one,a( is, ie+1 ),lda, rhs( 1 ), 1, one, c( ie+1, js ),1 )
						impl.Dgemv2('T', mb, m-ie, -1, d[is-1+ldd*(ie):], ldd, rhs[2:], 1, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemv( 'T', mb, m-ie, -one,d( is, ie+1 ),ldd , rhs( 3 ), 1, one, c( ie+1, js ),1 )

					}
				} else if (mb == 2) && (nb == 2) {
					impl.Dlaset(blas.All, ldz, ldz, 0, 0, z, ldz) //CALL dlaset( 'F', ldz, ldz, zero, zero, z, ldz )

					//z[0] = a[is-1+lda*(is-1)]    //z( 1, 1 ) = a( is, is ) T
					z[0] = a[is-1+lda*(is-1)]
					//z[1] = a[is-1+lda*(isp1-1)]  //z( 2, 1 ) = a( is, isp1 ) T
					z[8] = a[is-1+lda*(isp1-1)]
					//z[4] = -b[js-1+ldb*(js-1)]   //z( 5, 1 ) = -b( js, js ) T
					z[32] = -b[js-1+ldb*(js-1)]
					//z[6] = -b[jsp1-1+ldb*(js-1)] //z( 7, 1 ) = -b( jsp1, js ) T
					z[48] = -b[jsp1-1+ldb*(js-1)]

					//z[8] = a[isp1-1+lda*(is-1)]   //z( 1, 2 ) = a( isp1, is ) T
					z[1] = a[isp1-1+lda*(is-1)]
					//z[9] = a[isp1-1+lda*(isp1-1)] //z( 2, 2 ) = a( isp1, isp1 ) T
					z[9] = a[isp1-1+lda*(isp1-1)]
					//z[13] = -b[js-1+ldb*(js-1)]   //z( 6, 2 ) = -b( js, js ) T
					z[41] = -b[js-1+ldb*(js-1)]
					//z[15] = -b[jsp1-1+ldb*(js-1)] //z( 8, 2 ) = -b( jsp1, js ) T
					z[57] = -b[jsp1-1+ldb*(js-1)]
					//z[18] = a[is-1+lda*(is-1)]      //z( 3, 3 ) = a( is, is ) T
					z[18] = a[is-1+lda*(is-1)]
					//z[19] = a[is-1+lda*(isp1-1)]    //z( 4, 3 ) = a( is, isp1 ) T
					z[26] = a[is-1+lda*(isp1-1)]
					//z[20] = -b[js-1+ldb*(jsp1-1)]   //z( 5, 3 ) = -b( js, jsp1 ) T
					z[34] = -b[js-1+ldb*(jsp1-1)]
					//z[22] = -b[jsp1-1+ldb*(jsp1-1)] //z( 7, 3 ) = -b( jsp1, jsp1 ) T
					z[50] = -b[jsp1-1+ldb*(jsp1-1)]
					//z[26] = a[isp1-1+lda*(is-1)]    //z( 3, 4 ) = a( isp1, is ) T
					z[19] = a[isp1-1+lda*(is-1)]
					//z[27] = a[isp1-1+lda*(isp1-1)]  //z( 4, 4 ) = a( isp1, isp1 ) T
					z[27] = a[isp1-1+lda*(isp1-1)]
					//z[29] = -b[js-1+ldb*(jsp1-1)]   //z( 6, 4 ) = -b( js, jsp1 ) T
					z[43] = -b[js-1+ldb*(jsp1-1)]
					//z[31] = -b[jsp1-1+ldb*(jsp1-1)] //z( 8, 4 ) = -b( jsp1, jsp1 ) T
					z[59] = -b[jsp1-1+ldb*(jsp1-1)]
					//z[32] = d[is-1+ldd*(is-1)]   //z( 1, 5 ) = d( is, is ) T
					z[4] = d[is-1+ldd*(is-1)]
					//z[33] = d[is-1+ldd*(isp1-1)] //z( 2, 5 ) = d( is, isp1 ) T
					z[12] = d[is-1+ldd*(isp1-1)]
					//z[36] = -e[js-1+lde*(js-1)]  //z( 5, 5 ) = -e( js, js ) T
					z[36] = -e[js-1+lde*(js-1)]
					//z[41] = d[isp1-1+ldd*(isp1-1)] //z( 2, 6 ) = d( isp1, isp1 ) T
					z[13] = d[isp1-1+ldd*(isp1-1)]
					//z[45] = -e[js-1+lde*(js-1)]    //z( 6, 6 ) = -e( js, js ) T
					z[45] = -e[js-1+lde*(js-1)]
					//z[50] = d[is-1+ldd*(is-1)]    //z( 3, 7 ) = d( is, is ) T
					z[22] = d[is-1+ldd*(is-1)]
					//z[51] = d[is-1+ldd*(isp1-1)]  //z( 4, 7 ) = d( is, isp1 ) T
					z[30] = d[is-1+ldd*(isp1-1)]
					//z[52] = -e[js-1+lde*(jsp1-1)] //z( 5, 7 ) = -e( js, jsp1 ) T
					z[38] = -e[js-1+lde*(jsp1-1)]
					//z[54] = -e[jsp1-1+lde*(jsp1)] //z( 7, 7 ) = -e( jsp1, jsp1 ) T
					z[54] = -e[jsp1-1+lde*(jsp1)]
					//z[59] = d[isp1-1+ldd*(isp1-1)]  //z( 4, 8 ) = d( isp1, isp1 ) T
					z[31] = d[isp1-1+ldd*(isp1-1)]
					//z[61] = -e[js-1+lde*(jsp1-1)]   //z( 6, 8 ) = -e( js, jsp1 ) T
					z[47] = -e[js-1+lde*(jsp1-1)]
					//z[63] = -e[jsp1-1+lde*(jsp1-1)] //z( 8, 8 ) = -e( jsp1, jsp1 ) T
					z[63] = -e[jsp1-1+lde*(jsp1-1)]
					k = 1
					ii = mb*nb + 1
					for jj := 0; jj <= nb-1; jj++ { //DO 160 jj = 0, nb - 1
						bi.Dcopy(mb, c[is-1+ldc*(js+jj-1):], 1, rhs[k:], 1)
						//CALL dcopy( mb,           c( is, js+jj ), 1, rhs( k ), 1 )
						bi.Dcopy(mb, f[is-1+ldf*(js+jj-1):], 1, rhs[ii:], 1)
						//CALL dcopy( mb, f( is, js+jj ), 1, rhs( ii ), 1 )
						k = k + mb
						ii = ii + mb
					} //CONTINUE//G160:
					k = impl.Dgetc2(zdim, z, ldz, ipiv[:zdim], jpiv[:zdim]) //CALL dgetc2( zdim, z, ldz, ipiv, jpiv, ierr )
					if k > 0 {
						info = ierr
					}
					scaloc := impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim]) //impl.Dgesc2(zdim, z, ldz, rhs, ipiv[:zdim], jpiv[:zdim])//CALL dgesc2( zdim, z, ldz, rhs, ipiv, jpiv, scaloc )
					if scaloc != 1 {
						for k = 0; k < n; k++ { //DO 170 k = 1, n
							bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc) //CALL dscal( m, scaloc, c( 1, k ), 1 )
							bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf) //CALL dscal( m, scaloc, f( 1, k ), 1 )
						} //G170:
						scale = scale * scaloc
					}
					k = 1
					ii = mb*nb + 1
					for jj := 0; jj <= nb-1; jj++ { //DO 180 jj = 0, nb - 1
						bi.Dcopy(mb, rhs[k:], 1, c[is-1+ldc*(js+jj-1):], ldc)  //CALL dcopy( mb, rhs( k ), 1, c( is, js+jj ), 1 )
						bi.Dcopy(mb, rhs[ii:], 1, f[is-1+ldf*(js+jj-1):], ldf) //CALL dcopy( mb, rhs( ii ), 1, f( is, js+jj ), 1 )
						k = k + mb
						ii = ii + mb
					} //CONTINUE//G180:
					if j > p+2 {
						//bi.Dgemm(blas.NoTrans, blas.Trans, mb, js-1, nb, 1, c[is-1+ldc*(js-1):], ldc, b[ldb*(js-1):], ldb, 1, f[is-1:], ldf) //CALL dgemm('N','T', mb, js-1, nb, one,c( is, js ), ldc, b( 1, js ), ldb, one, f( is, 1 ), ldf )
						//bi.Dgemm(blas.NoTrans, blas.Trans, mb, js-1, nb, 1, f[is-1+ldf*(js-1):], ldf, e[lde*(js-1):], lde, 1, f[is-1:], ldf) //CALL dgemm('N','T', mb, js-1, nb, one,f( is, js ), ldf, e( 1, js ), lde, one,f( is, 1 ), ldf )
						impl.Dgemm2('N', 'T', mb, js-1, nb, 1, c[is-1+ldc*(js-1):], ldc, b[ldb*(js-1):], ldb, 1, f[is-1:], ldf) //CALL dgemm('N','T', mb, js-1, nb, one,c( is, js ), ldc, b( 1, js ), ldb, one, f( is, 1 ), ldf )
						impl.Dgemm2('N', 'T', mb, js-1, nb, 1, f[is-1+ldf*(js-1):], ldf, e[lde*(js-1):], lde, 1, f[is-1:], ldf) //CALL dgemm('N','T', mb, js-1, nb, one,f( is, js ), ldf, e( 1, js ), lde, one,f( is, 1 ), ldf )

					}
					if i < p {
						//bi.Dgemm(blas.Trans, blas.NoTrans, m-ie, nb, mb, -1, a[is-1+lda*(ie+1):], lda, c[is-1+ldc*(js):], ldc, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemm('T','N', m-ie, nb, mb, -one,a( is, ie+1 ), lda, c( is, js ), ldc, one, c( ie+1, js ), ldc )
						//bi.Dgemm(blas.Trans, blas.NoTrans, m-ie, nb, mb, -1, d[is-1+ldd*(ie+1):], ldd, f[is-1+ldf*(js):], ldf, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemm('T','N', m-ie, nb, mb, -one,d( is, ie+1 ), ldd, f( is, js ), ldf,one, c( ie+1, js ), ldc )
						impl.Dgemm2('T', 'N', m-ie, nb, mb, -1, a[is-1+lda*(ie+1):], lda, c[is-1+ldc*(js):], ldc, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemm('T','N', m-ie, nb, mb, -one,a( is, ie+1 ), lda, c( is, js ), ldc, one, c( ie+1, js ), ldc )
						impl.Dgemm2('T', 'N', m-ie, nb, mb, -1, d[is-1+ldd*(ie+1):], ldd, f[is-1+ldf*(js):], ldf, 1, c[ie+ldc*(js-1):], ldc) //CALL dgemm('T','N', m-ie, nb, mb, -one,d( is, ie+1 ), ldd, f( is, js ), ldf,one, c( ie+1, js ), ldc )

					}
				}

			} //CONTINUE//G190:
		} //CONTINUE//G200:
	}

	return scale, rdscal, rdsum, pq, info
}
