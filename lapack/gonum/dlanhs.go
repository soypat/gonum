// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
)

//import "gonum.org/v1/gonum/blas"
//DOUBLE PRECISION FUNCTION DLANHS( NORM, N, A, LDA, WORK )
func (impl Implementation) Dlanhs(norm byte, n int, a []float64, lda int, work []float64) (dlanhs float64) {
	//IMPLICIT NONE
	//  CHARACTER          norm
	//INTEGER            lda, n
	//DOUBLE PRECISION   a( lda, * ), work( * )
	//DOUBLE PRECISION   one, zero
	//parameter( one = 1.0d+0, zero = 0.0d+0 )
	var i, j int
	var sum, VALUE, scale float64
	_ = scale
	ssq := []float64{0, 0}
	colssq := []float64{0, 0}
	//LOGICAL            lsame, disnan
	//EXTERNAL           lsame, disnan
	//EXTERNAL           dlassq, dcombssq
	//INTRINSIC          abs, min, sqrt
	if n == 0 {
		VALUE = 0
	} else if norm == 'M' {
		VALUE = 0
		for j = 1; j < n; j++ { //DO 20 j = 1, n
			for i = 1; i <= min(n, j+1); i++ { //DO 10 i = 1, min( n, j+1 )
				sum = math.Abs(a[i-1+lda*(j-1)])
				if VALUE <= sum || math.IsNaN(sum) {
					VALUE = sum
				}
			} //10       CONTINUE
		} //20    CONTINUE
	} else if (norm == 'O') || (norm == '1') {
		VALUE = 0
		for j = 1; j <= n; j++ { //DO 40 j = 1, n
			sum = 0
			for i = 1; i <= min(n, j+1); i++ { //DO 30 i = 1, min( n, j+1 )
				sum = sum + math.Abs(a[i-1+lda*(j-1)])
			} //30       CONTINUE
			if VALUE < sum || math.IsNaN(sum) {
				VALUE = sum
			}
		} //40    CONTINUE
	} else if norm == 'I' {
		for i = 1; i <= n; i++ { //DO 50 i = 1, n
			work[i-1] = 0 //work( i ) = zero
		} //50    CONTINUE
		for j = 1; j <= n; j++ { //DO 70 j = 1, n
			for i = 1; i <= min(n, j+1); i++ { //DO 60 i = 1, min( n, j+1 )
				work[i-1] = work[i-1] + math.Abs(a[i-1+lda*(j-1)]) //work( i ) = work( i ) + abs( a( i, j ) )
			} //60       CONTINUE
		} //70    CONTINUE
		VALUE = 0
		for i = 1; i <= n; i++ { //DO 80 i = 1, n
			sum = work[i-1]
			if VALUE < sum || math.IsNaN(sum) {
				VALUE = sum
			}
		} //80    CONTINUE
	} else if (norm == 'F') || (norm == 'E') {
		ssq[0] = 0
		ssq[1] = 1
		for j = 1; j <= n; j++ { //DO 90 j = 1, n
			colssq[0] = 0
			colssq[1] = 1
			colssq[0], colssq[1] = impl.Dlassq(min(n, j+1), a[lda*(j-1):], 1, colssq[0], colssq[1]) //CALL dlassq( min( n, j+1 ), a( 1, j ), 1,                   colssq( 1 ), colssq( 2 ) )
			impl.Dcombssq(ssq, colssq)                                                              //CALL dcombssq( ssq, colssq )
		} //90    CONTINUE
		VALUE = ssq[0] * math.Sqrt(ssq[1])
	}
	dlanhs = VALUE
	return
}
