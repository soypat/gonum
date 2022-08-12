// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

//"math"

//import "gonum.org/v1/gonum/blas"
//DOUBLE PRECISION FUNCTION DLANHS( NORM, N, A, LDA, WORK )
func (impl Implementation) Dcombssq(v1, v2 []float64) {
	if v2[1] == 0 {
		return
	}
	if v1[0] >= v2[0] {
		if v1[0] != 0 {
			v1[1] = v1[1] + (v2[0]/v1[0])*(v2[0]/v1[0])*v2[1]
		} else {
			v1[1] = v1[1] + v2[1]
		}
	} else {
		v1[1] = v2[1] + (v1[0]/v2[0])*(v1[0]/v2[0])*v1[1]
		v1[0] = v2[0]
	}

}
