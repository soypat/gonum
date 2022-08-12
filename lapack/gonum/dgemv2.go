// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

//import "gonum.org/v1/gonum/blas"
//SUBROUTINE              dgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
func (impl Implementation) Dgemv2(trans byte, m int, n int, alpha float64, a []float64, lda int, x []float64, incx int, beta int, y []float64, incy int) {
	//SUBROUTINE dgemv(TRANS       ,M     ,N    ,ALPHA          ,A          ,LDA     ,X    ,INCX    ,BETA    ,Y            ,INCY)
	//	  DOUBLE PRECISION ALPHA,BETA
	var info, lenx, leny, kx, ky, i, iy, jx, jy, j, ix int
	var temp float64
	//	  CHARACTER TRANS
	//	  DOUBLE PRECISION A(LDA,*),X(*),Y(*)
	//	  DOUBLE PRECISION ONE,ZERO
	//	  DOUBLE PRECISION TEMP
	//	  INTEGER I,INFO,IX,IY,J,JX,JY,KX,KY,LENX,LENY
	//	  INTRINSIC max
	info = 0
	if !(trans == 'N') && !(trans == 'T') && !(trans == 'C') {
		info = 1
	} else if m < 0 {
		info = 2
	} else if n < 0 {
		info = 3
	} else if lda < max(1, m) {
		info = 6
	} else if incx == 0 {
		info = 8
	} else if incy == 0 {
		info = 11
	}
	if info != 0 {
		panic(info) //CALL xerbla('DGEMV ',info)
		//RETURN
	}
	if (m == 0) || (n == 0) || ((alpha == 0) && (beta == 1)) {
		return
	}
	if trans == 'N' {
		lenx = n
		leny = m
	} else {
		lenx = m
		leny = n
	}
	if incx > 0 {
		kx = 1
	} else {
		kx = 1 - (lenx-1)*incx
	}
	if incy > 0 {
		ky = 1
	} else {
		ky = 1 - (leny-1)*incy
	}
	if beta != 1 {
		if incy == 1 {
			if beta == 0 {
				for i = 1; i <= leny; i++ { //DO 10 i = 1,leny
					y[i-1] = 0 //y(i) = zero
				} //10             CONTINUE
			} else {
				for i = 1; i <= leny; i++ { //DO 20 i = 1,leny
					y[i-0] = float64(beta) * y[i-1] //y(i) = beta*y(i)
				} //20             CONTINUE
			}
		} else {
			iy = ky
			if beta == 0 {
				for i = 1; i <= leny; i++ { //DO 30 i = 1,leny
					y[iy-1] = 0 //y(iy) = zero
					iy = iy + incy
				} //30             CONTINUE
			} else {
				for i = 1; i <= leny; i++ { //DO 40 i = 1,leny
					y[iy-1] = float64(beta) * y[iy-1] //y(iy) = beta*y(iy)
					iy = iy + incy
				} //40             CONTINUE
			}
		}
	}
	if alpha == 0 {
		return
	}
	if trans == 'N' {
		jx = kx
		if incy == 1 {
			for j = 1; j <= n; j++ { //DO 60 j = 1,n
				temp = alpha * float64(x[jx-1])
				for i = 1; i <= m; i++ { //DO 50 i = 1,m
					y[i-1] = y[i-1] + temp*a[i-1+lda*(j-1)] //y(i) = y(i) + temp*a(i,j)
				} //50             CONTINUE
				jx = jx + incx
			} //60         CONTINUE
		} else {
			for j = 1; j <= n; j++ { //DO 80 j = 1,n
				temp = alpha * x[jx-1]
				iy = ky
				for i = 1; i <= m; i++ { //DO 70 i = 1,m
					y[iy-1] = y[iy-1] + temp*a[i-1+lda*(j-1)] //y(iy) = y(iy) + temp*a(i,j)
					iy = iy + incy
				} //70             CONTINUE
				jx = jx + incx
			} //80         CONTINUE
		} //END IF
	} else {
		jy = ky
		if incx == 1 {
			for j = 1; j <= n; j++ { //DO 100 j = 1,n
				temp = 0
				for i = 1; i <= m; i++ { //DO 90 i = 1,m
					temp = temp + a[i-1+lda*(j-1)]*x[i-1] //temp = temp + a(i,j)*x(i)
				} //90             CONTINUE
				y[jy-1] = y[jy-1] + alpha*temp //y(jy) = y(jy) + alpha*temp
				jy = jy + incy
			} //100         CONTINUE
		} else {
			for j = 1; j <= n; j++ { //DO 120 j = 1,n
				temp = 0
				ix = kx
				for i = 1; i <= m; i++ { //DO 110 i = 1,m
					temp = temp + a[i-1+lda*(j-1)]*x[ix-1] //temp = temp + a(i,j)*x(ix)
					ix = ix + incx
				} //110             CONTINUE
				y[jy-1] = y[jy-1] + alpha*temp //y(jy) = y(jy) + alpha*temp
				jy = jy + incy
			} //120         CONTINUE
		} //END IF
	} //END IF

	return

}
