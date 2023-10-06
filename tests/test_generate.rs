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




    let input_arr_cpu: Vec<f32> = vec![4.1, 1.7, -0.9,
                                        0.3, -2.0, 5.0,
                                        -1.0, 0.2, -3.1,
                                        9.0, -4.0, -6.2,
                                        0.3, 9.9, -5.1,
                                        -7.2, 0.6, -3.8,
                                        3.4, 2.0, 2.7];
	let mut input_arr = arrayfire::Array::new(&input_arr_cpu, arrayfire::Dim4::new(&[3, 7, 1, 1]));

    input_arr = arrayfire::transpose(&input_arr, false);
    arrayfire::print_gen("input_arr".to_string(), &input_arr, Some(6));

    let repeat_num = 5;

    RayBNN_Raytrace::Generate::Ray::tileDown(repeat_num, &mut input_arr);


}
