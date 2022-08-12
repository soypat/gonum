// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
)

// Dlarf applies an elementary reflector H to an m×n matrix C:
//  C = H * C  if side == blas.Left
//  C = C * H  if side == blas.Right
// H is represented in the form
//  H = I - tau * v * vᵀ
// where tau is a scalar and v is a vector.
//
// work must have length at least m if side == blas.Left and
// at least n if side == blas.Right.
//
// Dlarf is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dlarf2(side blas.Side, m, n int, v []float64, incv int, tau float64, c []float64, ldc int, work []float64) {
	var applyleft bool
	var lastc, lastv, i int
	applyleft = (side == blas.Left)
	lastv = 0
	lastc = 0
	if tau != 0 {
		if applyleft {
			lastv = m
		} else {
			lastv = n
		}
		if incv > 0 {
			i = 1 + (lastv-1)*incv
		} else {
			i = 1
		}
		for lastv > 0 && v[i-1] == 0 { //DO WHILE( lastv.GT.0 .AND. v( i ).EQ.zero )
			lastv = lastv - 1
			i = i - incv
		}
		if applyleft {
			lastc = impl.Iladlc(lastv, n, c, ldc) //lastc = iladlc(lastv, n, c, ldc)
			lastc = lastc + 1
		} else {
			lastc = impl.Iladlr(m, lastv, c, ldc) //lastc = iladlr(m, lastv, c, ldc)
			lastc = lastc + 1
		}
	}
	if applyleft {
		if lastv > 0 {
			//CALL dgemv( 'Transpose', lastv, lastc, one, c, ldc, v, incv,zero, work, 1 )
			impl.Dgemv2('T', lastv, lastc, 1, c, ldc, v, incv, 0, work, 1)
			//CALL dger( lastv, lastc, -tau, v, incv, work, 1, c, ldc )
			impl.Dger2(lastv, lastc, -tau, v, incv, work, 1, c, ldc)
		}
	} else {
		if lastv > 0 {
			//CALL dgemv( 'No transpose', lastc, lastv, one, c, ldc,v, incv, zero, work, 1 )
			impl.Dgemv2('N', lastc, lastv, 1, c, ldc, v, incv, 0, work, 1)
			//CALL dger( lastc, lastv, -tau, work, 1, v, incv, c, ldc )
			impl.Dger2(lastc, lastv, -tau, work, 1, v, incv, c, ldc)
		}
	}
	return
}
