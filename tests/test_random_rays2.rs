#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_random_rays2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


	arrayfire::set_seed(1231);






}
