

use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use RayBNN_Sparse::Util::Convert::get_global_weight_idx;



use crate::Generate::Fixed::tileDown;

use crate::Intersect::Sphere::line_sphere_intersect_batchV2;

use crate::Generate::Fixed::rays_from_neuronsA_to_neuronsB;





const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const EPSILON_F64: f64 = 1.0e-3;

const ONEMINUSEPSILON_F64: f64 = ONE_F64 - EPSILON_F64;

const RAYTRACE_LIMIT: u64 = 100000000;













