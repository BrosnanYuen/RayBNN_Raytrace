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
    //arrayfire::print_gen("input_arr".to_string(), &input_arr, Some(6));

    let repeat_num = 5;

    assert_eq!(input_arr.dims()[0], 7);
    assert_eq!(input_arr.dims()[1], 3);

    RayBNN_Raytrace::Generate::Fixed::tileDown(repeat_num, &mut input_arr);

    assert_eq!(input_arr.dims()[0], repeat_num*7);
    assert_eq!(input_arr.dims()[1], 3);

    //arrayfire::print_gen("input_arr".to_string(), &input_arr, Some(6));

    let mut input_arr_repeat = vec!(f32::default();input_arr.elements());
	input_arr.host(&mut input_arr_repeat);


    let input_arr_act: Vec<f32> = vec![4.1, 4.1, 4.1, 4.1, 4.1, 0.3, 0.3, 0.3, 0.3, 0.3, -1.0, -1.0, -1.0, -1.0, -1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.3, 0.3, 0.3, 0.3, 0.3, -7.2, -7.2, -7.2, -7.2, -7.2, 3.4, 3.4, 3.4, 3.4, 3.4, 1.7, 1.7, 1.7, 1.7, 1.7, -2.0, -2.0, -2.0, -2.0, -2.0, 0.2, 0.2, 0.2, 0.2, 0.2, -4.0, -4.0, -4.0, -4.0, -4.0, 9.9, 9.9, 9.9, 9.9, 9.9, 0.6, 0.6, 0.6, 0.6, 0.6, 2.0, 2.0, 2.0, 2.0, 2.0, -0.9, -0.9, -0.9, -0.9, -0.9, 5.0, 5.0, 5.0, 5.0, 5.0, -3.1, -3.1, -3.1, -3.1, -3.1, -6.2, -6.2, -6.2, -6.2, -6.2, -5.1, -5.1, -5.1, -5.1, -5.1, -3.8, -3.8, -3.8, -3.8, -3.8, 2.7, 2.7, 2.7, 2.7, 2.7];

    assert_eq!(input_arr_act, input_arr_repeat);














    let input_arr_cpu: Vec<f64> = vec![4.1, 1.7, -0.9,
                                        0.3, -2.0, 5.0,
                                        -1.0, 0.2, -3.1,
                                        9.0, -4.0, -6.2,
                                        0.3, 9.9, -5.1,
                                        -7.2, 0.6, -3.8,
                                        3.4, 2.0, 2.7];
	let mut input_arr = arrayfire::Array::new(&input_arr_cpu, arrayfire::Dim4::new(&[3, 7, 1, 1]));

    input_arr = arrayfire::transpose(&input_arr, false);
    //arrayfire::print_gen("input_arr".to_string(), &input_arr, Some(6));

    let repeat_num = 5;

    assert_eq!(input_arr.dims()[0], 7);
    assert_eq!(input_arr.dims()[1], 3);

    RayBNN_Raytrace::Generate::Fixed::tileDown(repeat_num, &mut input_arr);

    assert_eq!(input_arr.dims()[0], repeat_num*7);
    assert_eq!(input_arr.dims()[1], 3);

    //arrayfire::print_gen("input_arr".to_string(), &input_arr, Some(6));

    let mut input_arr_repeat = vec!(f64::default();input_arr.elements());
	input_arr.host(&mut input_arr_repeat);


    let input_arr_act: Vec<f64> = vec![4.1, 4.1, 4.1, 4.1, 4.1, 0.3, 0.3, 0.3, 0.3, 0.3, -1.0, -1.0, -1.0, -1.0, -1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.3, 0.3, 0.3, 0.3, 0.3, -7.2, -7.2, -7.2, -7.2, -7.2, 3.4, 3.4, 3.4, 3.4, 3.4, 1.7, 1.7, 1.7, 1.7, 1.7, -2.0, -2.0, -2.0, -2.0, -2.0, 0.2, 0.2, 0.2, 0.2, 0.2, -4.0, -4.0, -4.0, -4.0, -4.0, 9.9, 9.9, 9.9, 9.9, 9.9, 0.6, 0.6, 0.6, 0.6, 0.6, 2.0, 2.0, 2.0, 2.0, 2.0, -0.9, -0.9, -0.9, -0.9, -0.9, 5.0, 5.0, 5.0, 5.0, 5.0, -3.1, -3.1, -3.1, -3.1, -3.1, -6.2, -6.2, -6.2, -6.2, -6.2, -5.1, -5.1, -5.1, -5.1, -5.1, -3.8, -3.8, -3.8, -3.8, -3.8, 2.7, 2.7, 2.7, 2.7, 2.7];

    assert_eq!(input_arr_act, input_arr_repeat);

















    let input_arr_cpu: Vec<i32> = vec![41, 17, -09,
                                        03, -20, 50,
                                        -10, 02, -31,
                                        90, -40, -62,
                                        03, 99, -51,
                                        -72, 06, -38,
                                        34, 20, 27];
	let mut input_arr = arrayfire::Array::new(&input_arr_cpu, arrayfire::Dim4::new(&[3, 7, 1, 1]));

    input_arr = arrayfire::transpose(&input_arr, false);
    //arrayfire::print_gen("input_arr"to_string(), &input_arr, Some(6));

    let repeat_num = 5;

    assert_eq!(input_arr.dims()[0], 7);
    assert_eq!(input_arr.dims()[1], 3);

    RayBNN_Raytrace::Generate::Fixed::tileDown(repeat_num, &mut input_arr);

    assert_eq!(input_arr.dims()[0], repeat_num*7);
    assert_eq!(input_arr.dims()[1], 3);

    //arrayfire::print_gen("input_arr"to_string(), &input_arr, Some(6));

    let mut input_arr_repeat = vec!(i32::default();input_arr.elements());
	input_arr.host(&mut input_arr_repeat);


    let input_arr_act: Vec<i32> = vec![41, 41, 41, 41, 41, 03, 03, 03, 03, 03, -10, -10, -10, -10, -10, 90, 90, 90, 90, 90, 03, 03, 03, 03, 03, -72, -72, -72, -72, -72, 34, 34, 34, 34, 34, 17, 17, 17, 17, 17, -20, -20, -20, -20, -20, 02, 02, 02, 02, 02, -40, -40, -40, -40, -40, 99, 99, 99, 99, 99, 06, 06, 06, 06, 06, 20, 20, 20, 20, 20, -09, -09, -09, -09, -09, 50, 50, 50, 50, 50, -31, -31, -31, -31, -31, -62, -62, -62, -62, -62, -51, -51, -51, -51, -51, -38, -38, -38, -38, -38, 27, 27, 27, 27, 27];

    assert_eq!(input_arr_act, input_arr_repeat);




}
