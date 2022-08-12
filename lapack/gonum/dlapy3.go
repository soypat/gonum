// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlapy2 is the LAPACK version of math.Hypot.
//
// Dlapy2 is an internal routine. It is exported for testing purposes.
func (Implementation) Dlapy3(x, y, z float64) float64 {
	var xabs, yabs, zabs, w, dlapy3 float64
	xabs = math.Abs(x)
	yabs = math.Abs(y)
	zabs = math.Abs(z)
	w = math.Max(math.Max(xabs, yabs), zabs)
	if w == 0 {

		dlapy3 = xabs + yabs + zabs
	} else {
		dlapy3 = w * math.Sqrt((xabs/w)*(xabs/w)+(yabs/w)*(yabs/w)+(zabs/w)*(zabs/w))
	}
	return dlapy3
}
