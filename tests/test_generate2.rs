#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_generate() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let con_rad = 6.2;

    RayBNN_Raytrace::Generate::Ray::filter_rays(
        con_rad,
    
        target_input_pos: &arrayfire::Array<Z>,
    
        input_pos: &mut arrayfire::Array<Z>,
        input_idx: &mut arrayfire::Array<i32>,
    );




}
