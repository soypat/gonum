// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

//import "gonum.org/v1/gonum/blas"
//SUBROUTINE              dgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
func (impl Implementation) Dgemm2(transa byte, transb byte, m int, n int, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {

	//	  DOUBLE PRECISION ALPHA,BETA
	//INTEGER K,LDA,LDB,LDC,M,N
	// CHARACTER TRANSA,TRANSB
	// DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
	//  LOGICAL LSAME
	// EXTERNAL lsame
	// EXTERNAL xerbla
	//  INTRINSIC max
	var temp float64
	var i, info, j, l, nrowa, nrowb int //INTEGER I,INFO,J,L,NROWA,NROWB
	var nota, notb bool
	//DOUBLE PRECISION ONE,ZERO
	//parameter(one=1.0d+0,zero=0.0d+0)
	_ = nota
	_ = notb
	nota = (transa == 'N')
	notb = (transb == 'N')
	if nota {
		nrowa = m
	} else {
		nrowa = k
	}
	if notb {
		nrowb = k
	} else {
		nrowb = n
	}
	info = 0
	if (!nota) && (!(transa == 'C')) && (!(transa == 'T')) {
		info = 1
	} else if (!notb) && (!(transb == 'C')) && (!(transb == 'T')) {
		info = 2
	} else if m < 0 {
		info = 3
	} else if n < 0 {
		info = 4
	} else if k < 0 {
		info = 5
	} else if lda < max(1, nrowa) {
		info = 8
	} else if ldb < max(1, nrowb) {
		info = 10
	} else if ldc < max(1, m) {
		info = 13
	}
	if info != 0 {
		panic(info)
		//return
	}
	if (m == 0) || (n == 0) || (((alpha == 0) || (k == 0)) && (beta == 1)) {
		return
	}
	if alpha == 0 {
		if beta == 0 {
			for j = 1; j <= n; j++ { //DO 20 j = 1,n
				for i = 1; i <= m; i++ { //DO 10 i = 1,m
					c[i-1+ldc*(j-1)] = 0 //c(i,j) = zero
				} //10             CONTINUE
			} //20         CONTINUE
		} else {
			for j = 1; j <= n; j++ { //DO 40 j = 1,n
				for i = 1; i <= m; i++ { //DO 30 i = 1,m
					c[i-1+ldc*(j-1)] = beta * c[i-1+ldc*(j-1)] //c(i,j) = beta*c(i,j)
				} //30             CONTINUE
			} //40         CONTINUE
		}
		return
	}
	if notb {
		if nota {
			for j = 1; j <= n; j++ { //DO 90 j = 1,n
				if beta == 0 {
					for i := 1; i <= m; i++ { //DO 50 i = 1,m
						c[i-1+ldc*(j-1)] = 0 //c(i,j) = zero
					} //50                 CONTINUE

				} else if beta != 1 {
					for i = 1; i <= m; i++ { //DO 60 i = 1,m
						c[i-1+ldc*(j-1)] = beta * c[i-1+ldc*(j-1)] //c(i,j) = beta*c(i,j)
					} //60                 CONTINUE
				} //END IF
				for l = 1; l <= k; l++ { //DO 80 l = 1,k
					temp = alpha * b[l-1+ldb*(j-1)] //temp = alpha*b(l,j)
					for i = 1; i <= m; i++ {        //DO 70 i = 1,m
						c[i-1+ldc*(j-1)] = c[i-1+ldc*(j-1)] + temp*a[i-1+lda*(l-1)] //c(i,j) = c(i,j) + temp*a(i,l)
					} //70                 CONTINUE
				} //80             CONTINUE
			} //90         CONTINUE
		} else {
			for j = 1; j <= n; j++ { //DO 120 j = 1,n
				for i = 1; i <= m; i++ { //DO 110 i = 1,m
					temp = 0
					for l = 1; l <= k; l++ { //DO 100 l = 1,k
						temp = temp + a[l-1+lda*(i-1)]*b[l-1+ldb*(j-1)] ///temp = temp + a(l,i)*b(l,j)
					} //100                 CONTINUE
					if beta == 0 {
						c[i-1+ldc*(j-1)] = alpha * temp //c(i,j) = alpha*temp
					} else {
						c[i-1+ldc*(j-1)] = alpha*temp + beta*c[i-1+ldc*(j-1)] //c(i,j) = alpha*temp + beta*c(i,j)
					} //END IF
				} //110             CONTINUE
			} //120         CONTINUE
		}
	} else {
		if nota {
			for j = 1; j < n; j++ { //DO 170 j = 1,n
				if beta == 0 {
					for i = 1; i <= m; i++ { //DO 130 i = 1,m
						c[i-1+ldc*(j-1)] = 0 //c(i,j) = zero
					} //130                 CONTINUE
				} else if beta != 1 {
					for i = 1; i <= m; i++ { //DO 140 i = 1,m
						c[i-1+ldc*(j-1)] = beta * c[i-1+ldc*(j-1)] //c(i,j) = beta*c(i,j)
					} //140                 CONTINUE
				} //END IF
				for l = 1; l <= k; l++ { //DO 160 l = 1,k
					temp = alpha * b[j-1+ldb*(l-1)] //temp = alpha*b(j,l)
					for i = 1; i <= m; i++ {        //DO 150 i = 1,m
						c[i-1+ldc*(j-1)] = c[i-1+ldc*(j-1)] + temp*a[i-1+lda*(l-1)] //c(i,j) = c(i,j) + temp*a(i,l)
					} //150                 CONTINUE
				} //160             CONTINUE
			} //170         CONTINUE
		} else {
			for j = 1; j <= n; j++ { //DO 200 j = 1,n
				for i = 1; i <= m; i++ { //DO 190 i = 1,m
					temp = 0
					for l = 1; l <= k; l++ { //DO 180 l = 1,k
						temp = temp + a[l-1+lda*(i-1)]*b[j-1+ldb*(l-1)] //temp = temp + a(l,i)*b(j,l)
					} //180                 CONTINUE
					if beta == 0 {
						c[i-1+ldc*(j-1)] = alpha * temp //c(i,j) = alpha*temp
					} else {
						c[i-1+ldc*(j-1)] = alpha*temp + beta*c[i-1+ldc*(j-1)] //c(i,j) = alpha*temp + beta*c(i,j)
					} //END IF
				} //190             CONTINUE
			} //200         CONTINUE
		} //END IF
	} //END IF
	return
}
