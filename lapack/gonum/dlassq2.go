// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
)

//import "gonum.org/v1/gonum/blas"
//SUBROUTINE              dgemm(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
func (impl Implementation) Dlassq2(n int, x []float64, incx int, scl float64, sumsq float64) (sclout float64, sumsqout float64) {
	//subroutine dlassq( n, x, incx, scl, sumsq )
	//    use la_constants, &
	//       only: wp=>dp, zero=>dzero, one=>done, &
	//             sbig=>dsbig, ssml=>dssml, tbig=>dtbig, tsml=>dtsml
	//    use la_xisnan
	//    integer :: incx, n
	//    real(wp) :: scl, sumsq
	//    real(wp) :: x(*)
	var i, ix int
	var notbig bool
	var abig, amed, asml, ax, ymax, ymin, tbig, sbig, tsml, ssml float64

	//if( (scl==nil) || (sumsq==nil) ){
	//	return
	//}
	if sumsq == 0 {
		scl = 1
	}
	if scl == 0 {
		scl = 1
		sumsq = 0
	}
	if n <= 0 {
		return
	}
	notbig = true
	asml = 0
	amed = 0
	abig = 0
	ix = 1
	if incx < 0 {
		ix = 1 - (n-1)*incx
	}
	for i = 1; i <= n; i++ { //do i = 1, n
		ax = math.Abs(x[ix-1])
		if ax > tbig {
			abig = abig + (ax*sbig)*(ax*sbig)
			notbig = false
		} else if ax < tsml {
			if notbig {
				asml = asml + (ax*ssml)*(ax*ssml)
			}
		} else {
			amed = amed + ax*ax
		}
		ix = ix + incx
	} //end do
	if sumsq > 0 {
		ax = scl * math.Sqrt(sumsq)
		if ax > tbig {
			abig = abig + (scl*sbig)*(scl*sbig)*sumsq
		} else if ax < tsml {
			if notbig {
				asml = asml + (scl*ssml)*(scl*ssml)*sumsq
			}
		} else {
			amed = amed + scl*scl*sumsq
		}
	}
	if abig > 0 {
		if amed > 0 { //if (amed > 0 || (amed==nil)) {
			abig = abig + (amed*sbig)*sbig
		}
		scl = 1 / sbig
		sumsq = abig
	} else if asml > 0 {
		if amed > 0 { //if (amed > 0 || (amed==nil)) {
			amed = math.Sqrt(amed)
			asml = math.Sqrt(asml) / ssml
			if asml > amed {
				ymin = amed
				ymax = asml
			} else {
				ymin = asml
				ymax = amed
			}
			scl = 1
			sumsq = ymax * ymax * (1 + (ymin/ymax)*(ymin/ymax))
		} else {
			scl = 1 / ssml
			sumsq = asml
		}
	} else {
		scl = 1
		sumsq = amed
	}
	return scl, sumsq
}
