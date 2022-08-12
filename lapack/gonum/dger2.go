// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

//import "gonum.org/v1/gonum/blas"
//SUBROUTINE              dgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
func (impl Implementation) Dger2(m int, n int, alpha float64, x []float64, incx int, y []float64, incy int, a []float64, lda int) {
	//SUBROUTINE dger(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
	//	  DOUBLE PRECISION ALPHA
	//	  INTEGER INCX,INCY,LDA,M,N
	//	  DOUBLE PRECISION A(LDA,*),X(*),Y(*)
	//	  DOUBLE PRECISION TEMP
	var i, info, ix, j, jy, kx int
	var temp float64
	info = 0
	if m < 0 {
		info = 1
	} else if n < 0 {
		info = 2
	} else if incx == 0 {
		info = 5
	} else if incy == 0 {
		info = 7
	} else if lda < max(1, m) {
		info = 9
	}
	if info != 0 {
		panic(info)
	}
	if (m == 0) || (n == 0) || (alpha == 0) {
		return
	}
	if incy > 0 {
		jy = 1
	} else {
		jy = 1 - (n-1)*incy
	}
	if incx == 1 {
		for j = 1; j <= n; j++ { //DO 20 j = 1,n
			if y[jy-1] != 0 {
				temp = alpha * y[jy-1]
				for i = 1; i <= m; i++ { //DO 10 i = 1,m
					a[i-1+lda*(j-1)] = a[i-1+lda*(j-1)] + x[i-1]*temp //a(i,j) = a(i,j) + x(i)*temp
				} //10             CONTINUE
			}
			jy = jy + incy
		} //20     CONTINUE
	} else {
		if incx > 0 {
			kx = 1
		} else {
			kx = 1 - (m-1)*incx
		} //END IF
		for j = 1; j <= n; j++ { //DO 40 j = 1,n
			if y[jy-1] != 0 {
				temp = alpha * y[jy-1]
				ix = kx
				for i = 1; i <= m; i++ { //DO 30 i = 1,m
					a[i-1+lda*(j-1)] = a[i-1+lda*(j-1)] + x[ix-1]*temp //a(i,j) = a(i,j) + x(ix)*temp
					ix = ix + incx
				} //30             CONTINUE
			}
			jy = jy + incy
		} //40     CONTINUE
	}

	return

}
